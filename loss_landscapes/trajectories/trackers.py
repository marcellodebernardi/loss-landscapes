"""
Classes and functions for tracking a model's optimization trajectory and computing
a low-dimensional approximation of the trajectory.
"""


from abc import ABC, abstractmethod
import numpy as np
from loss_landscapes.model_interface.model_agnostic_factories import wrap_model


class TrajectoryTracker(ABC):
    """
    A TrajectoryTracker facilitates tracking the optimization trajectory of a
    DL/RL model. Trajectory trackers provide facilities for storing model parameters
    as well as for retrieving and operating on stored parameters.
    """
    def __init__(self):
        # trajectory is a list of ParameterVector objects
        self.trajectory = []

    @abstractmethod
    def __getitem__(self, timestep) -> np.ndarray:
        """
        Returns the position of the model from the given training timestep as a numpy array.
        :param timestep: training step of parameters to retrieve
        :return: numpy array
        """
        pass

    @abstractmethod
    def get_item(self, timestep) -> np.ndarray:
        """
        Returns the position of the model from the given training timestep as a numpy array.
        :param timestep: training step of parameters to retrieve
        :return: numpy array
        """
        pass

    @abstractmethod
    def save_position(self, model):
        """
        Appends the current model parameterization to the stored training trajectory.
        :param model: model object with current state of interest
        :return: N/A
        """
        pass

    def get_trajectory(self) -> list:
        """
        Returns a reference to the currently stored trajectory.
        :return: numpy array
        """
        return self.trajectory


class FullTrajectoryTracker(TrajectoryTracker):
    """
    A FullTrajectoryTracker is a tracker which stores a history of points in the tracked
    model's original parameter space, and can be used to perform a variety of computations
    on the trajectory.

    WARNING: be aware that full trajectory tracking requires N * M memory, where N is the
    number of iterations tracked and M is the size of the model. The amount of memory used
    by the trajectory tracker can easily become very large.
    """
    def __init__(self, model):
        super().__init__()
        self.trajectory.append(wrap_model(model).get_parameters(deepcopy=True).as_numpy())

    def __getitem__(self, timestep) -> np.ndarray:
        return self.trajectory[timestep]

    def get_item(self, timestep) -> np.ndarray:
        return self.__getitem__(timestep)

    def save_position(self, model):
        self.trajectory.append(wrap_model(model).get_parameters(deepcopy=True).as_numpy())


class ProjectingTrajectoryTracker(TrajectoryTracker):
    """
    A ProjectingTrajectoryTracker is a tracker which applies dimensionality reduction to
    all model parameterizations upon storage. This is particularly appropriate for large
    models, where storing a history of points in the model's parameter space would be
    unfeasible in terms of memory.
    """
    def __init__(self, model, n_bases=2):
        super().__init__()

        n = wrap_model(model).get_parameters().numel()
        self.A = np.column_stack(
            [np.random.normal(size=n) for _ in range(n_bases)]
        )

    def __getitem__(self, timestep) -> np.ndarray:
        return self.trajectory[timestep]

    def get_item(self, timestep) -> np.ndarray:
        return self.__getitem__(timestep)

    def save_position(self, model):
        # we solve the equation Ax = b using least squares, where A is the matrix of basis vectors
        b = wrap_model(model).get_parameters().as_numpy()
        self.trajectory.append(np.linalg.lstsq(self.A, b, rcond=None)[0])
