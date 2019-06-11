import inspect
import loss_landscapes.model_interface.model_wrapper as model_wrapper
import loss_landscapes.model_interface.torch.torch_wrappers as torch_wrappers


SUPPORTED_LIBRARIES = {
    'torch': torch_wrappers.TorchModelWrapper
}
SUPPORTED_MODEL_TYPES = {
    'torch.nn.modules.module.Module': torch_wrappers.TorchModelWrapper,
}


class AgentInterface:
    """ Defines how to operate on an agent object that isn't just a neural network. """
    def __init__(self, library: str, components: list = None, layers: list = None, call_fn: callable = None):
        """
        Define an agent interface.
        :param library: string flag indicating which DL library the agent uses
        :param components: list of models to be considered by the library
        :param call_fn: function of the form forward(model, x) that calls the model on input x
        :param layers: list of layers to be included - if none, all layers included
        """
        if library not in list(SUPPORTED_LIBRARIES.keys()):
            raise ValueError('Unrecognized library flag. Supported libraries: %s' % list(SUPPORTED_LIBRARIES.keys()))
        self.library = library
        self.components = components
        self.layers = layers
        self.call_fn = call_fn

    def get_library(self):
        return self.library

    def get_components(self):
        return self.components

    def get_layers(self):
        return self.layers

    def get_forward_fn(self):
        return self.call_fn


def wrap_model(model, agent_interface=None) -> model_wrapper.ModelWrapper:
    """
    Returns an appropriate wrapper for the given model. For example, if the
    model is a PyTorch model, returns a TorchModelWrapper for the model.
    :param model: model to wrap
    :param agent_interface: defines how to access components etc for complex agents
    :return: appropriate wrapper for model
    """
    if isinstance(model, model_wrapper.ModelWrapper):
        return model

    if agent_interface is not None:
        components = agent_interface.get_components()
        layers = agent_interface.get_layers()
        forward_fn = agent_interface.get_forward_fn()
        library = agent_interface.get_library()
    else:
        components = None
        layers = None
        forward_fn = None
        library = None

    # assume straightforward model
    try:
        model_type = _identify_model_type(model)
        return SUPPORTED_MODEL_TYPES[model_type](model, components, layers, forward_fn)
    except TypeError:
        if agent_interface is not None:
            model_type = library
            return SUPPORTED_LIBRARIES[model_type](model, components, layers, forward_fn)
        else:
            raise ValueError('Unrecognized model type. AgentInterface must be provided for unrecognized model types.')


def _identify_model_type(obj):
    type_hierarchy = [c.__module__ + '.' + c.__name__ for c in inspect.getmro(type(obj))]

    # if any of the supported model types are a match, import and return corresponding DL library
    for model_type in list(SUPPORTED_MODEL_TYPES.keys()):
        if model_type in type_hierarchy:
            return model_type

    # if not a supported model type, give up
    raise TypeError('Unrecognized model type.')
