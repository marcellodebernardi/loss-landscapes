from abc import ABC, abstractmethod


class Evaluator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model):
        pass


class EvaluatorPipeline(Evaluator):
    def __init__(self, evaluators: list):
        super().__init__()
        self.evaluators = evaluators

    def __call__(self, model) -> tuple:
        return tuple([eval_f(model) for eval_f in self.evaluators])
