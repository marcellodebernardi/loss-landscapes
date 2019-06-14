"""
Defines functions for getting and setting the parameters of a model.
"""

import copy
import itertools
import loss_landscapes.model_interface.model_wrapper as model_wrapper
import loss_landscapes.model_interface.torch.torch_tensor as torch_tensor


class TorchModelWrapper(model_wrapper.ModelWrapper):
    def __init__(self, model, components=None, layers=None, call_fn=None):
        super().__init__(model, components, layers, call_fn)
        # if user leaves components=None, we assume the entire model is the only component
        self.components = [model] if self.components is None else self.components

        # each tuple identifies a specific parameter; this list defines the ordering of ParameterTensors
        self.parameter_names = [(module_index, parameter_name)
                                for module_index, module in enumerate(self.components, 0)
                                for parameter_name, param in module.named_parameters()
                                if self.layers is None or param in self.layers
                                ]

    def __call__(self, x):
        if self.forward_fn is not None:
            return self.forward_fn(self.model, x)
        else:
            return self.model(x)

    def get_parameter_tensor(self, deepcopy=False) -> 'torch_tensor.TorchParameterTensor':
        """
        Return a TorchParameterTensor wrapping the named parameters of the underlying model.
        The parameters can either be returned as a view of the model parameters or as a copy.
        :param deepcopy: whether to view or deepcopy the model parameters
        :return: view or deepcopy of accessible model parameters
        """
        parameters = []

        # construct a 2D list indexed by [module, parameter]
        for i, module in enumerate(self.components, 0):
            state_dict = module.state_dict()

            for j, name in self.parameter_names:
                if i == j:
                    parameters.append(state_dict[name].clone().detach() if deepcopy else state_dict[name])

        return torch_tensor.TorchParameterTensor(parameters)

    def set_parameter_tensor(self, new_parameters: torch_tensor.TorchParameterTensor):
        """
        Sets the parameters of the wrapped model to the given ParameterVector.
        :param new_parameters: ParameterVector of new parameters
        :return: none
        """
        for i, module in enumerate(self.components, 0):
            new_state_dict = copy.deepcopy(module.state_dict())  # todo deepcopy unnecessary, assign directly to s_dict?
            # find all parameters pertaining to current module
            for j, param_name in self.parameter_names:
                if i == j:
                    new_state_dict[param_name] = new_parameters[j]
            # load new state dictionary
            module.load_state_dict(new_state_dict)

    def parameters(self):
        raise NotImplementedError('Only named parameters are exposed.')

    def named_parameters(self):
        return itertools.chain(p
                               for c in self.components
                               for _, p in c.named_parameters()
                               if self.layers is None or p in self.layers
                               )

    def zero_grad(self):
        for component in self.components:
            component.zero_grad()
