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
from loss_landscapes.model_interface.model_parameters import rand_u_like
from loss_landscapes.model_interface.model_wrapper import ModelWrapper


class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()


class LossGradient(Metric):
    """ Computes the gradient of a specified loss function w.r.t. the model parameters
    over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> np.ndarray:
        loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target)
        gradient = torch.autograd.grad(loss, model_wrapper.named_parameters()).detach().numpy()
        model_wrapper.zero_grad()
        return gradient


class LossPerturbations(Metric):
    """ Computes random perturbations in the loss value along a sample or random directions.
    These perturbations can be used to reason probabilistically about the curvature of a
    point on the loss landscape, as demonstrated in the paper by Schuurmans et al
    (https://arxiv.org/abs/1811.11214)."""
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor, n_directions, alpha):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target
        self.n_directions = n_directions
        self.alpha = alpha

    def __call__(self, model_wrapper: ModelWrapper) -> np.ndarray:
        # start point and directions
        start_point = model_wrapper.get_module_parameters()
        start_loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()

        # compute start loss and perturbed losses
        results = []
        for idx in range(self.n_directions):
            direction = rand_u_like(start_point)
            start_point.add_(direction)

            loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()
            results.append(loss - start_loss)

            start_point.sub_(direction)

        return np.array(results)


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
