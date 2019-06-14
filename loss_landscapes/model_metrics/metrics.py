from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model):
        pass


class MetricPipeline(Metric, ABC):
    def __init__(self, evaluators: list):
        super().__init__()
        self.evaluators = evaluators

    @abstractmethod
    def __call__(self, model) -> tuple:
        pass
