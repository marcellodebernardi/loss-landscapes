import inspect
import loss_landscapes.common.model_interface.model_wrapper as model_wrapper
import loss_landscapes.common.model_interface.torch.torch_wrappers as torch_wrappers


SUPPORTED_MODEL_TYPES = {
    'torch.nn.modules.module.Module': torch_wrappers.TorchModelWrapper,
}


def wrap_model(model, forward_fn=None) -> model_wrapper.ModelWrapper:
    """
    Returns an appropriate wrapper for the given model. For example, if the
    model is a PyTorch model, returns a TorchModelWrapper for the model.
    :param model: model to wrap
    :param forward_fn: function for obtaining model output
    :return: appropriate wrapper for model
    """
    model_type = _identify_dl_library(model, recursion_depth=1)
    return SUPPORTED_MODEL_TYPES[model_type](model, forward_fn)


def _identify_dl_library(obj, recursion_depth=1):
    type_hierarchy = [c.__module__ + '.' + c.__name__ for c in inspect.getmro(type(obj))]

    # if any of the supported model types are a match, import and return corresponding DL library
    for model_type in list(SUPPORTED_MODEL_TYPES.keys()):
        if model_type in type_hierarchy:
            return model_type

    # if max recursion depth reached before type is identified, give up
    if recursion_depth != 0:
        # try all the attributes of the object
        for o in dir(obj):
            try:
                return _identify_dl_library(o, recursion_depth - 1)
            except TypeError:
                pass

    # if recursion depth reached or no attributes provided identification, give up
    raise TypeError('Unrecognized model type.')
