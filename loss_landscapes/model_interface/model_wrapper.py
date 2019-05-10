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
        self.torch_keys = sorted(self.model.state_dict().keys())

        # remove keys referring to persistent buffers and other non-parameter contents
        for idx in range(len(self.torch_keys) - 1, -1, -1):
            key = self.torch_keys[idx]
            if not isinstance(self.model.state_dict()[key], torch.nn.parameter.Parameter):
                self.torch_keys.pop(idx)

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
        for key in self.torch_keys:
            parameters.append(copy.deepcopy(self.model.state_dict()[key]))
        return ParameterVector(parameters)

    def set_parameters(self, new_parameters: ParameterVector):
        """
        Sets the model's parameters using the given list of parameters.
        :param new_parameters: list of numpy arrays
        :return: none
        """
        new_state_dict = dict()
        for i in range(len(new_parameters)):
            new_state_dict[self.torch_keys[i]] = copy.deepcopy(new_parameters[i])
        self.model.load_state_dict(new_state_dict)
