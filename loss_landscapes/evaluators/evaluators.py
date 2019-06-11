from abc import ABC, abstractmethod


class Evaluator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model):
        pass


class EvaluatorPipeline(Evaluator, ABC):
    def __init__(self, evaluators: list):
        super().__init__()
        self.evaluators = evaluators

    @abstractmethod
    def __call__(self, model) -> tuple:
        pass
