"""
A library of pre-written metric evaluators for more general use cases than loss functions.

The evaluators in this module are not specific to the landscape of the training loss, and
can be useful in other situations; for example, a ClassAccuracyEvaluator can be used to
inspect the classification accuracy of a model under different parameter value assignments.
"""

from abc import ABC, abstractmethod
import torch.nn
from loss_landscapes.model_metrics.metrics import Metric


class TorchMetric(Metric, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model: torch.nn.Module):
        pass


class ClassificationAccuracyEvaluator(TorchMetric):
    """ Computes the model's classification accuracy over a specified set of inputs-labels pairs. """
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = inputs
        self.targets = labels
        self.n = len(labels)

    def __call__(self, model: torch.nn.Module) -> float:
        predictions = torch.argmax(model(self.inputs), dim=1)
        return 100 * (predictions.int() == self.targets.int()).sum().item() / self.n
