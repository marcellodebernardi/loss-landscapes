from abc import ABC, abstractmethod
from loss_landscapes.common.evaluators.evaluators import Evaluator, EvaluatorPipeline


class SLEvaluator(Evaluator, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model):
        pass


class SLEvaluatorPipeline(EvaluatorPipeline, ABC):
    def __init__(self, evaluators: list):
        super().__init__(evaluators)

    @abstractmethod
    def __call__(self, model) -> tuple:
        pass
