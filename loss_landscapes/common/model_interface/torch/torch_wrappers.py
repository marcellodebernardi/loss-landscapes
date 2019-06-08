"""
Defines functions for getting and setting the parameters of a model.
"""

import copy
import inspect
import torch.nn
import loss_landscapes.common.model_interface.model_wrapper as model_wrapper
import loss_landscapes.common.model_interface.torch.torch_tensor as torch_tensor


class TorchModelWrapper(model_wrapper.ModelWrapper):
    def __init__(self, model, get_param_fn=None, forward_fn=None):
        """
        Construct a model wrapper which only exposes and operates on the underlying
        model's parameters (torch.nn.Module.parameters()).

        :param model: model to wrap
        """
        super().__init__(model, get_param_fn, forward_fn)
        self.modules = self.get_components_fn() if self.get_components_fn is not None else self.model

        # each tuple identifies a specific parameter; this list defines the ordering of ParameterTensors
        self.parameter_names = [(module_index, parameter_name)
                                for module_index, module in enumerate(self.modules, 0)
                                for parameter_name, _ in module.named_parameters()
                                ]

    def forward(self, x):
        if self.forward_fn is not None:
            return self.forward_fn(self.model, x)
        else:
            return self.model(x)

    def get_parameters(self, deepcopy=False) -> 'torch_tensor.TorchParameterTensor':
        """
        Return a TorchParameterTensor wrapping the named parameters of the underlying model.
        The parameters can either be returned as a view of the model parameters or as a copy.
        :param deepcopy: whether to view or deepcopy the model parameters
        :return: view or deepcopy of accessible model parameters
        """
        parameters = []

        # construct a 2D list indexed by [module, parameter]
        for i, module in enumerate(self.modules, 0):
            state_dict = module.state_dict()

            for j, name in self.parameter_names:
                if i == j:
                    parameters.append(state_dict[name].clone().detach() if deepcopy else state_dict[name])

        return torch_tensor.TorchParameterTensor(parameters)

    def set_parameters(self, new_parameters: torch_tensor.TorchParameterTensor):
        """
        Sets the parameters of the wrapped model to the given ParameterVector.
        :param new_parameters: ParameterVector of new parameters
        :return: none
        """
        for i, module in enumerate(self.modules, 0):
            new_state_dict = copy.deepcopy(module.state_dict())  # todo deepcopy unnecessary, assign directly to s_dict?
            # find all parameters pertaining to current module
            for j, param_name in self.parameter_names:
                if i == j:
                    new_state_dict[param_name] = new_parameters[j]
            # load new state dictionary
            module.load_state_dict(new_state_dict)
