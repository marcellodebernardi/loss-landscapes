"""
Utilities to support the library when dealing with external inputs.

The functions defined in this module are for internal use only. They are utilities
for dealing with the different object types used by different numerical computation
libraries like PyTorch and TensorFlow.
"""


import torch
import torch.nn


SUPPORTED_LIBRARIES = ['pytorch']
UNRECOGNIZED_LIBRARY_MSG = 'Could not determine the model\'s numerical computation library of origin. ' \
                           + 'The library or library version is unsupported, or the models ' \
                           + 'were passed incorrectly to loss-landscapes. ' \
                           + 'Supported libraries include ' + str(SUPPORTED_LIBRARIES) + '. '\
                           + 'Check the documentation for how to correctly pass your ' \
                           + 'model to loss-landscapes.'


def determine_library(*args) -> str:
    """
    Returns a string flag representing the numerical computation library from which the
    model and the parameters have originated. Returns None if undetermined.

    Args:
        args: any number of model objects for which we want to determine the library of origin

    Returns:
        str: 'torch' if models are from PyTorch, and so on
    """
    if isinstance(args[0], torch.nn.Module):
        for model in args:
            if not isinstance(model, torch.nn.Module):
                raise ValueError(UNRECOGNIZED_LIBRARY_MSG + 'Expected torch.nn.Module, got ' + str(type(model)))
        return 'torch'
    else:
        raise ValueError(UNRECOGNIZED_LIBRARY_MSG)


def _is_torch_state(parameters) -> bool:
    """
    Returns true if the given parameters are a list of torch.nn.Parameter objects.

    Args:
        parameters: should be a list of torch.nn.Parameter objects

    Returns:
        bool: True if input is a list of torch.nn.Parameter objects
    """
    if not isinstance(parameters, list):
        return False
    for p in parameters:
        if not isinstance(p, torch.nn.Parameter):
            return False
    return True
