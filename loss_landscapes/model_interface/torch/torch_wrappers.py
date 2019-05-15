"""
Defines functions for getting and setting the parameters of a model.
"""

import copy
from abc import ABC

import torch
import torch.nn
import loss_landscapes.model_interface.model_wrapper as model_wrapper
import loss_landscapes.model_interface.torch.torch_tensor as torch_tensor


class TorchModelWrapper(model_wrapper.ModelWrapper, ABC):
    """ A ModelWrapper which hides a PyTorch model, defined as any subclass of torch.nn.Module. """
    def __init__(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise ValueError('Model of type ' + str(type(model)) + ' does not inherit torch.nn.Module.')
        self.model = model


class TorchNamedParameterWrapper(TorchModelWrapper):
    """
    A TorchModelWrapper which enables manipulating only the model's named parameters.
    Persistent buffers and unnamed parameters are invisible to wrappers of this class.
    """
    def __init__(self, model: torch.nn.Module, layer_names=None):
        """
        Construct a model wrapper which only exposes and operates on the underlying
        model's named parameters (torch.nn.Module.named_parameters()). By default
        the wrapper considers all of the model's named parameters, but can be limited
        to specific layers

        :param model: model to wrap
        :param layer_names:
        """
        super().__init__(model)
        # sorted parameter names are used to define a 'correct' ordering
        # of tensors for generated ParameterVector objects.
        self.parameter_names = sorted([name for name, _ in self.model.named_parameters()])

        # if user indicates specific layers, then only keep those layers
        if layer_names is not None:
            for name in self.parameter_names:
                if name not in layer_names:
                    self.parameter_names.remove(name)

    def get_model(self) -> torch.nn.Module:
        """
        Returns a reference to the PyTorch model wrapped by this NamedParameterWrapper.
        :return: wrapped model
        """
        return self.model

    def get_parameters(self) -> 'torch_tensor.TorchParameterTensor':
        """
        Return a deep copy of the parameters made accessible by this wrapper.
        :return: deep copy of accessible model parameters
        """
        parameters = []
        # use keys from stored key list to ensure list is consistently ordered
        state_dict = self.model.state_dict()
        for name in self.parameter_names:
            parameters.append(state_dict[name].clone().detach().double())
        return torch_tensor.TorchParameterTensor(parameters)

    def set_parameters(self, new_parameters: torch_tensor.TorchParameterTensor):
        """
        Sets the parameters of the wrapped model to the given ParameterVector.
        :param new_parameters: ParameterVector of new parameters
        :return: none
        """
        new_state_dict = copy.deepcopy(self.model.state_dict())
        for idx, p in enumerate(new_parameters, 0):
            new_state_dict[self.parameter_names[idx]] = copy.deepcopy(p)
        self.model.load_state_dict(new_state_dict)
