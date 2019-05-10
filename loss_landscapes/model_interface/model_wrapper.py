"""
Defines functions for getting and setting the parameters of a model.
"""

import copy
import torch
import torch.nn
from loss_landscapes.model_interface.parameter_vector import ParameterVector


class ModelWrapper:
    """ Wraps a model and defines the operations for getting its weights or updating them. """
    def __init__(self, model):
        super().__init__()
        if not isinstance(model, torch.nn.Module):
            raise ValueError('Model is not a subclass of torch.nn.Module.')

        self.model = model
        self.names = []

        for name, _ in self.model.named_parameters():
            self.names.append(name)

        self.names.sort()

    def get_model(self):
        """
        Returns a reference to the model wrapped by this ModelInterface.
        :return: wrapped model
        """
        return self.model

    def build_parameter_vector(self) -> ParameterVector:
        """
        Returns the parameters of the model as a list of torch tensors.
        :return: list of numpy arrays
        """
        parameters = []
        # use keys from stored key list to ensure list is consistently ordered
        state_dict = self.model.state_dict()
        for name in self.names:
            parameters.append(copy.deepcopy(state_dict[name]))
        return ParameterVector(parameters)

    def set_parameters(self, new_parameters: ParameterVector):
        """
        Sets the model's parameters using the given list of parameters.
        :param new_parameters: list of numpy arrays
        :return: none
        """
        new_state_dict = copy.deepcopy(self.model.state_dict())
        for idx, p in enumerate(new_parameters, 0):
            new_state_dict[self.names[idx]] = copy.deepcopy(p)
        self.model.load_state_dict(new_state_dict)
