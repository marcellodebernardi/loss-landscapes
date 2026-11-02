"""
A library of pre-written evaluation functions for PyTorch loss functions.

The classes and functions in this module cover common loss landscape evaluations. In particular,
computing the loss, the gradient of the loss (w.r.t. model parameters) and Hessian of the loss
(w.r.t. model parameters) for some supervised learning loss is easily accomplished.
"""

import numpy as np
import torch
import torch.autograd

from loss_landscapes.metrics.metric import Metric
from loss_landscapes.model_interface.model_parameters import ModelParameters, rand_n_like
from loss_landscapes.model_interface.model_wrapper import ModelWrapper


class Loss(Metric):
    """Computes a specified loss function over specified input-output pairs."""

    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()


class LossGradient(Metric):
    """Computes the gradient of a specified loss function w.r.t. the model parameters
    over specified input-output pairs, returned as a flat array in parameter order."""

    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> np.ndarray:
        # wrap_model() disables gradient tracking so parameters can be modified in place,
        # so enable it only for this evaluation and restore the previous state afterwards
        parameters = list(model_wrapper.parameters())
        previous = [p.requires_grad for p in parameters]
        try:
            for p in parameters:
                p.requires_grad_(True)
            with torch.enable_grad():
                loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target)
                if not loss.requires_grad:
                    raise ValueError(
                        "The loss does not depend on any of the wrapped parameters. Check that the "
                        "wrapper exposes the modules its forward function uses, and that the forward "
                        "function does not detach them or run them under torch.no_grad()."
                    )
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        finally:
            for p, requires_grad in zip(parameters, previous, strict=True):
                p.requires_grad_(requires_grad)

        # parameters the loss does not depend on have a zero gradient, and sparse gradients
        # (e.g. from torch.nn.Embedding(sparse=True)) must be made dense to convert to numpy
        return ModelParameters(
            [
                torch.zeros_like(p) if g is None else g.detach().to_dense()
                for p, g in zip(parameters, gradients, strict=True)
            ]
        ).as_numpy()


class LossPerturbations(Metric):
    """Computes the change in loss when the model parameters are perturbed by a small step
    in either direction along each of a sample of random directions.

    For each direction d, drawn uniformly from the unit sphere, the loss is evaluated at
    theta + alpha * d and theta - alpha * d, and the changes from the loss at theta are
    returned as a pair (delta_plus, delta_minus). If the loss is approximately quadratic
    within distance alpha of theta, then delta_plus - delta_minus estimates 2 * alpha times
    the directional derivative along d, and delta_plus + delta_minus estimates alpha ** 2
    times the curvature along d. These can be used to reason probabilistically about the
    curvature of a point on the loss landscape, as demonstrated by Ahmed et al. (2019)
    (https://arxiv.org/abs/1811.11214).

    alpha is an absolute distance in parameter space, not a multiple of the parameters'
    norm, and must be small enough for the quadratic approximation to hold. The model is
    returned to its starting parameters, up to floating-point rounding, after each evaluation.
    """

    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor, n_directions, alpha):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target
        self.n_directions = n_directions
        self.alpha = alpha

    def __call__(self, model_wrapper: ModelWrapper) -> np.ndarray:
        """
        :return: array of shape (n_directions, 2), whose i-th row is (delta_plus, delta_minus)
            for the i-th random direction
        """
        start_point = model_wrapper.get_module_parameters()
        start_loss = self._loss(model_wrapper)

        results = []
        for _ in range(self.n_directions):
            # a Gaussian sample scaled to unit length is uniformly distributed on the unit sphere
            step = rand_n_like(start_point)
            step.mul_(self.alpha / step.model_norm())

            # multiple of step currently added to the start point, so that it can be undone
            # even if evaluating the loss raises
            applied = 0.0
            try:
                start_point.add_(step)
                applied = 1.0
                delta_plus = self._loss(model_wrapper) - start_loss
                start_point.sub_(step * 2.0)
                applied = -1.0
                delta_minus = self._loss(model_wrapper) - start_loss
            finally:
                start_point.sub_(step * applied)

            results.append((delta_plus, delta_minus))

        return np.array(results).reshape(self.n_directions, 2)

    def _loss(self, model_wrapper: ModelWrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()


# noinspection DuplicatedCode
# class GradientPredictivenessEvaluator(Metric):
#     """
#     Computes the L2 norm of the distance between loss gradients at consecutive
#     iterations. We consider a gradient to be predictive if a move in the direction
#     of the gradient results in a similar gradient at the next step; that is, the
#     gradients of the loss change smoothly along the optimization trajectory.
#
#     This evaluator is inspired by experiments ran by Santurkar et al (2018), for
#     details see https://arxiv.org/abs/1805.11604
#     """
#     def __init__(self, supervised_loss_fn, inputs, target):
#         super().__init__(None, None, None)
#         self.gradient_evaluator = GradientEvaluator(supervised_loss_fn, inputs, target)
#         self.previous_gradient = None
#
#     def __call__(self, model) -> float:
#         if self.previous_gradient is None:
#             self.previous_gradient = self.gradient_evaluator(model)
#             return 0.0
#         else:
#             current_grad = self.gradient_evaluator(model)
#             previous_grad = self.previous_gradient
#             self.previous_gradient = current_grad
#             # return l2 distance of current and previous gradients
#             return np.linalg.norm(current_grad - previous_grad, ord=2)
#
#
# class BetaSmoothnessEvaluator(Metric):
#     """
#     Computes the "beta-smoothness" of the gradients, as characterized by
#     Santurkar et al (2018). The beta-smoothness of a function at any given point
#     is the ratio of the magnitude of the change in its gradients, over the magnitude
#     of the change in input. In the case of loss landscapes, it is the ratio of the
#     magnitude of the change in loss gradients over the magnitude of the change in
#     parameters. In general, we call a function f beta-smooth if
#
#         |f'(x) - f'(y)| < beta|x - y|
#
#     i.e. if there exists an upper bound beta on the ratio between change in gradients
#     and change in input. Santurkar et al call "effective beta-smoothness" the maximum
#     encountered ratio along some optimization trajectory.
#
#     This evaluator is inspired by experiments ran by Santurkar et al (2018), for
#     details see https://arxiv.org/abs/1805.11604
#     """
#
#     def __init__(self, supervised_loss_fn, inputs, target):
#         super().__init__(None, None, None)
#         self.gradient_evaluator = GradientEvaluator(supervised_loss_fn, inputs, target)
#         self.previous_gradient = None
#         self.previous_parameters = None
#
#     def __call__(self, model):
#         if self.previous_parameters is None:
#             self.previous_gradient = self.gradient_evaluator(model)
#             self.previous_parameters = TorchModelWrapper(model).get_parameter_tensor().numpy()
#             return 0.0
#         else:
#             current_grad = self.gradient_evaluator(model)
#             current_p = TorchModelWrapper(model).get_parameter_tensor().numpy()
#             previous_grad = self.previous_gradient
#             previous_p = self.previous_parameters
#
#             self.previous_gradient = current_grad
#             self.previous_parameters = current_p
#             # return l2 distance of current and previous gradients
#             return np.linalg.norm(current_grad - previous_grad, ord=2) / np.linalg.norm(current_p - previous_p, ord=2)


# todo - these are complicated by the fact that hessian matrix is of size O(n^2) in the number of NN params
# ideally there would be a way to compute the eigenvalues incrementally, without computing the whole hessian
# matrix first.

# class HessianEvaluator(SupervisedTorchEvaluator):
#     """
#     Computes the Hessian of a specified loss function w.r.t. the model
#     parameters over specified input-output pairs.
#     """
#     def __init__(self, supervised_loss_fn, inputs, target):
#         super().__init__(supervised_loss_fn, inputs, target)
#
#     def __call__(self, model) -> np.ndarray:
#         loss = self.loss_fn(model(self.inputs), self.target)
#         gradient = torch.autograd.grad(loss, [p for _, p in model.named_parameters()], create_graph=True)
#         gradient = torch.cat(tuple([p.view(-1) for p in gradient]))
#         numel = sum([param.numel() for param in gradient])
#
#         # for computing higher-order gradients, see https://github.com/pytorch/pytorch/releases/tag/v0.2.0
#         hessian = torch.zeros(size=(numel, numel))
#
#         for derivative, idx in enumerate(gradient, 0):
#             hessian[idx] = torch.autograd.grad(torch.tensor(derivative), [p.view(-1) for _, p in model.named_parameters()])
#
#         return hessian.detach().numpy()
#
#
# class PrincipalCurvaturesEvaluator(SupervisedTorchEvaluator):
#     """
#     Computes the principal curvatures of a specified loss function over
#     specified input-output pairs. The principal curvatures are the
#     eigenvalues of the Hessian matrix.
#     """
#     def __init__(self, supervised_loss_fn, inputs, target):
#         super().__init__(None, None, None)
#         self.hessian_evaluator = HessianEvaluator(supervised_loss_fn, inputs, target)
#
#     def __call__(self, model) -> np.ndarray:
#         return np.linalg.eigvals(self.hessian_evaluator(model))
#
#
# class CurvaturePositivityEvaluator(SupervisedTorchEvaluator):
#     """
#     Computes the extent of the positivity of a loss function's curvature at a
#     specific point in parameter space. The extent of positivity is measured as
#     the fraction of dimensions with positive curvature. Optionally, dimensions
#     can be weighted by the magnitude of their curvature.
#
#     Inspired by a related metric in the paper by Li et al,
#     http://papers.nips.cc/paper/7875-visualizing-the-loss-landscape-of-neural-nets.
#     """
#     def __init__(self, supervised_loss_fn, inputs, target, weighted=False):
#         super().__init__(None, None, None)
#         self.curvatures_evaluator = PrincipalCurvaturesEvaluator(supervised_loss_fn, inputs, target)
#         self.weighted = weighted
#
#     def __call__(self, model) -> np.ndarray:
#         curvatures = self.curvatures_evaluator(model)
#         # ratio of sum of all positive curvatures over sum of all negative curvatures
#         if self.weighted:
#             positive_total = curvatures[(curvatures >= 0)].sum()
#             negative_total = np.abs(curvatures[(curvatures < 0)].sum())
#             return positive_total / negative_total
#         # fraction of dimensions with positive curvature
#         else:
#             return np.array((curvatures >= 0).sum() / curvatures.size())
