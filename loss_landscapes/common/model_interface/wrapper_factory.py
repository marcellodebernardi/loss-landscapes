import inspect
import loss_landscapes.common.model_interface.model_wrapper as model_wrapper
import loss_landscapes.common.model_interface.torch.torch_wrappers as torch_wrappers


SUPPORTED_MODEL_TYPES = {
    'torch.nn.modules.module.Module': torch_wrappers.TorchModelWrapper,
}


def wrap_model(model, agent_interface=None) -> model_wrapper.ModelWrapper:
    """
    Returns an appropriate wrapper for the given model. For example, if the
    model is a PyTorch model, returns a TorchModelWrapper for the model.
    :param model: model to wrap
    :param agent_interface: defines how to access components etc for complex agents
    :return: appropriate wrapper for model
    """
    try:
        model_type = _identify_model_type(model)
        return SUPPORTED_MODEL_TYPES[model_type](model)
    except TypeError:
        if agent_interface is not None:
            get_components_fn, forward_fn = agent_interface.get_configuration()
            return SUPPORTED_MODEL_TYPES[model_type](model, get_components_fn, forward_fn)


def _identify_model_type(obj):
    type_hierarchy = [c.__module__ + '.' + c.__name__ for c in inspect.getmro(type(obj))]

    # if any of the supported model types are a match, import and return corresponding DL library
    for model_type in list(SUPPORTED_MODEL_TYPES.keys()):
        if model_type in type_hierarchy:
            return model_type

    # if not a supported model type, give up
    raise TypeError('Unrecognized model type.')
