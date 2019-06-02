import inspect


def _import_torch():
    import torch as t
    return t


def _import_tensorflow():
    import tensorflow as tf
    return tf


def _import_chainer():
    import chainer as c
    return c


SUPPORTED_MODEL_TYPES = {
    'torch.nn.modules.module.Module': _import_torch,
}


def import_dl_library(obj, recursion_depth=1):
    type_hierarchy = [c.__module__ + '.' + c.__name__ for c in inspect.getmro(type(obj))]

    # if any of the supported model types are a match, import and return corresponding DL library
    for model_type in list(SUPPORTED_MODEL_TYPES.keys()):
        if model_type in type_hierarchy:
            return SUPPORTED_MODEL_TYPES[model_type]()

    # if max recursion depth reached before type is identified, give up
    if recursion_depth != 0:
        # try all the attributes of the object
        for o in dir(obj):
            try:
                return import_dl_library(o, recursion_depth - 1)
            except TypeError:
                pass

    # if recursion depth reached or no attributes provided identification, give up
    raise TypeError('Unrecognized model type.')





