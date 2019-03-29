import numpy as np
from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def line(self, model, direction, distance, steps, evaluation_f, agent=None, model_set_f=None) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def plane(self, model, direction_one, direction_two, distance_one, distance_two, steps, evaluation_f, center=False,
              agent=None, model_set_f=None) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def sample_uniform_like(self, source_parameters, unit_vector=True):
        raise NotImplementedError()

    @abstractmethod
    def get_normalized_vector(self, direction, model_parameters, norm_type):
        raise NotImplementedError()
