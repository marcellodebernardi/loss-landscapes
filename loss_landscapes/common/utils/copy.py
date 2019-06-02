"""
Utilities for safely copying models and avoid aliasing.

Several client-facing methods of the loss-landscapes library require passing a reference to one or
more model objects. For example, a user wanting to compute a linear interpolation plot of the model
loss between the model initialization and the trained model has to store a copy of the model before
training. The copy of the model in its initial state must be a deep copy, entirely detached from
the original model, in order to avoid the risk of aliasing problems. Aliasing of parameters in
particular is to be avoided.

It is not always clear what a simple and effective method of deep copying a model might be. The simple
copy.deepcopy is not a robust approach, as it is not necessarily supported by every version of every
DL library, and the manner in which it is supported is not necessarily consistent between libraries
and versions.

Deep copying of models and avoiding aliasing issues is left as the responsibility of the client code
using the loss-landscapes library. This module provides (optional) model copying utilities. Due to
the problems mentioned above, and the volatility of DL library APIs, they might not work.
"""
import copy


def _import_torch():
    # on-demand import for torch
    import torch as t
    return t


LIBRARY_IMPORTS = {
    'torch': _import_torch
}


def deepcopy_model(model, library):
    # import the relevant modules of the user's DL library
    modules = LIBRARY_IMPORTS[library]()

    if library == 'torch':
        return _copy_pytorch_module(model, modules)


def _copy_pytorch_module(model, t):
    original_state_dict = copy.deepcopy(model.state_dict())

    new_model = copy.deepcopy(model)
    new_model.load_state_dict(original_state_dict)
    return new_model
