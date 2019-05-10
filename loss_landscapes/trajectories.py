

import numpy as np
import model_interface.parameter_vector as pv
from model_interface.parameter_vector import ParameterVector
from model_interface.model_wrapper import ModelWrapper


class TrajectoryTracker:
    def __init__(self):
        # trajectory is a list of ParameterVector objects
        self.trajectory = []

    def __getitem__(self, timestep) -> ParameterVector:
        """
        Returns the model parameters from the given training timestep as a ParameterVector.
        :param timestep: training step of parameters to retrieve
        :return: ParameterVector
        """
        return self.trajectory[timestep]

    def save_position(self, model):
        """
        Appends the current model parameterization to the stored training trajectory.
        :param model: model object with current state of interest
        :return: N/A
        """
        self.trajectory.append(ModelWrapper(model).build_parameter_vector())

    def get_trajectory(self) -> np.ndarray:
        """
        Returns a reference to the currently stored trajectory.
        :return: numpy array
        """
        return np.stack([p.numpy() for p in self.trajectory])

    def get_reduced_trajectory(self, directions='random') -> np.ndarray:
        coordinates = []

        # todo is sampling each parameter independently equivalent to sampling direction uniformly?
        if directions == 'random':
            dir_x = pv.rand_u_like(self.trajectory[0])
            dir_y = pv.rand_u_like(self.trajectory[0])
        elif directions == 'pca':
            raise NotImplementedError('PCA not yet implemented.')
        else:
            raise NotImplementedError('Unsupported direction selection method.')

        for parameter_vector in self.trajectory:
            # solve system of equations Ax = b ... where A is n*2 matrix containing dir_x and dir_y
            # and b is the resulting parameter vector. x is therefore the coefficients of the linear
            # combination c_1 * dir_x + c_2 * dir_y = parameter_vector.
            A = np.transpose(np.stack([dir_x, dir_y]))
            b = parameter_vector.numpy()
            x = np.linalg.lstsq(A, b)
            coordinates.append(x)

        return np.array(coordinates)
