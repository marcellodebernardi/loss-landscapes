import inspect
import loss_landscapes.common.model_interface.model_wrapper as model_wrapper
import loss_landscapes.common.model_interface.torch.torch_wrappers as torch_wrappers


SUPPORTED_LIBRARIES = {
    'torch': torch_wrappers.TorchModelWrapper
}
SUPPORTED_MODEL_TYPES = {
    'torch.nn.modules.module.Module': torch_wrappers.TorchModelWrapper,
}


class AgentInterface:
    """ Defines how to operate on an agent object that isn't just a neural network. """
    def __init__(self, get_components_fn, forward_fn, library):
        """
        Define an agent interface.
        :param get_components_fn: function that returns a list of models pertaining to DL library is use
        :param forward_fn: function of the form forward(model, x) that calls the model on input x
        :param library: string flag indicating which DL library the agent uses
        """
        if library not in list(SUPPORTED_LIBRARIES.keys()):
            raise ValueError('Unrecognized library flag. Supported libraries: %s' % list(SUPPORTED_LIBRARIES.keys()))
        self.library = library
        self.get_components_fn = get_components_fn
        self.forward_fn = forward_fn

    def get_library(self):
        return self.library

    def get_configuration(self):
        return self.get_components_fn, self.forward_fn


def wrap_model(model, agent_interface) -> model_wrapper.ModelWrapper:
    """
    Returns an appropriate wrapper for the given model. For example, if the
    model is a PyTorch model, returns a TorchModelWrapper for the model.
    :param model: model to wrap
    :param agent_interface: defines how to access components etc for complex agents
    :return: appropriate wrapper for model
    """
    try:
        model_type = _identify_model_type(model)
        return SUPPORTED_MODEL_TYPES[model_type](model, None, None)
    except TypeError:
        if agent_interface is not None:
            library = agent_interface.get_library()
            get_components_fn, forward_fn = agent_interface.get_configuration()
            return SUPPORTED_MODEL_TYPES[library](model, get_components_fn, forward_fn)
        else:
            raise ValueError('AgentInterface must be provided for unrecognized model types.')


def _identify_model_type(obj):
    type_hierarchy = [c.__module__ + '.' + c.__name__ for c in inspect.getmro(type(obj))]

    # if any of the supported model types are a match, import and return corresponding DL library
    for model_type in list(SUPPORTED_MODEL_TYPES.keys()):
        if model_type in type_hierarchy:
            return model_type

    # if not a supported model type, give up
    raise TypeError('Unrecognized model type.')
