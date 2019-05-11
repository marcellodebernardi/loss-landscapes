"""
A library of pre-written evaluation functions for PyTorch loss functions.

The classes and functions in this module cover common loss landscape evaluations. In particular,
computing the loss, the gradient of the loss (w.r.t. model parameters) and Hessian of the loss
(w.r.t. model parameters) for some supervised learning loss is easily accomplished.
"""


from abc import ABC, abstractmethod
import numpy as np
import torch.autograd
from loss_landscapes.evaluators.evaluator import Evaluator
from loss_landscapes.model_interface.torch.torch_wrappers import NamedParameterWrapper


class SupervisedTorchEvaluator(Evaluator, ABC):
    """ Abstract class for PyTorch supervised learning evaluation functions. """
    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__()
        self.loss_fn = supervised_loss_fn
        self.inputs = inputs
        self.target = target

    @abstractmethod
    def __call__(self, model):
        pass


class LossEvaluator(SupervisedTorchEvaluator):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__(supervised_loss_fn, inputs, target)

    def __call__(self, model) -> np.ndarray:
        return np.array(self.loss_fn(model(self.inputs), self.target))


class GradientEvaluator(SupervisedTorchEvaluator):
    """
    Computes the gradient of a specified loss function w.r.t. the model parameters
    over specified input-output pairs.
    """
    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__(supervised_loss_fn, inputs, target)

    def __call__(self, model) -> np.ndarray:
        loss = self.loss_fn(model(self.inputs), self.target)
        # for computing higher-order gradients, see https://github.com/pytorch/pytorch/releases/tag/v0.2.0
        gradient = torch.autograd.grad(loss, model.parameters())
        return gradient.detach().numpy()


class HessianEvaluator(SupervisedTorchEvaluator):
    """
    Computes the Hessian of a specified loss function w.r.t. the model
    parameters over specified input-output pairs.
    """
    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__(supervised_loss_fn, inputs, target)

    def __call__(self, model) -> np.ndarray:
        loss = self.loss_fn(model(self.inputs), self.target)
        # for computing higher-order gradients, see https://github.com/pytorch/pytorch/releases/tag/v0.2.0
        gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        hessian = torch.autograd.grad(gradient, model.parameters())
        return hessian.detach().numpy()


class PrincipalCurvaturesEvaluator(SupervisedTorchEvaluator):
    """
    Computes the principal curvatures of a specified loss function over
    specified input-output pairs. The principal curvatures are the
    eigenvalues of the Hessian matrix.
    """
    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__(None, None, None)
        self.hessian_evaluator = HessianEvaluator(supervised_loss_fn, inputs, target)

    def __call__(self, model) -> np.ndarray:
        return np.linalg.eigvals(self.hessian_evaluator(model))


class GradientPredictivenessEvaluator(SupervisedTorchEvaluator):
    """
    Computes the L2 norm of the distance between loss gradients at consecutive
    iterations. We consider a gradient to be predictive if a move in the direction
    of the gradient results in a similar gradient at the next step; that is, the
    gradients of the loss change smoothly along the optimization trajectory.

    This evaluator is inspired by experiments ran by Santurkar et al (2018), for
    details see https://arxiv.org/abs/1805.11604
    """
    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__(None, None, None)
        self.gradient_evaluator = GradientEvaluator(supervised_loss_fn, inputs, target)
        self.previous_gradient = None

    def __call__(self, model) -> float:
        if self.previous_gradient is None:
            self.previous_gradient = self.gradient_evaluator(model)
            return 0.0
        else:
            current_grad = self.gradient_evaluator(model)
            previous_grad = self.previous_gradient
            self.previous_gradient = current_grad
            # return l2 distance of current and previous gradients
            return np.linalg.norm(current_grad - previous_grad, ord=2)


class BetaSmoothnessEvaluator(SupervisedTorchEvaluator):
    """
    Computes the "beta-smoothness" of the gradients, as characterized by
    Santurkar et al (2018). The beta-smoothness of a function at any given point
    is the ratio of the magnitude of the change in its gradients, over the magnitude
    of the change in input. In the case of loss landscapes, it is the ratio of the
    magnitude of the change in loss gradients over the magnitude of the change in
    parameters. In general, we call a function f beta-smooth if

        |f'(x) - f'(y)| < beta|x - y|

    i.e. if there exists an upper bound beta on the ratio between change in gradients
    and change in input. Santurkar et al call "effective beta-smoothness" the maximum
    encountered ratio along some optimization trajectory.

    This evaluator is inspired by experiments ran by Santurkar et al (2018), for
    details see https://arxiv.org/abs/1805.11604
    """

    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__(None, None, None)
        self.gradient_evaluator = GradientEvaluator(supervised_loss_fn, inputs, target)
        self.previous_gradient = None
        self.previous_parameters = None

    def __call__(self, model):
        if self.previous_parameters is None:
            self.previous_gradient = self.gradient_evaluator(model)
            self.previous_parameters = NamedParameterWrapper(model).get_parameters().numpy()
            return 0.0
        else:
            current_grad = self.gradient_evaluator(model)
            current_p = NamedParameterWrapper(model).get_parameters().numpy()
            previous_grad = self.previous_gradient
            previous_p = self.previous_parameters

            self.previous_gradient = current_grad
            self.previous_parameters = current_p
            # return l2 distance of current and previous gradients
            return np.linalg.norm(current_grad - previous_grad, ord=2) / np.linalg.norm(current_p - previous_p, ord=2)


class PlateauEvaluator(SupervisedTorchEvaluator):
    """
    Evaluator that computes the ratio between the change in loss and the change in parameters.
    Large changes in parameters with little change in loss indicates a plateau
    """
    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__(supervised_loss_fn, inputs, target)

    def __call__(self, model):
        # todo how to best characterize a plateau?
        raise NotImplementedError()
