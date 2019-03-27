import torch
import torch.nn

SUPPORTED_LIBRARIES = ['pytorch']
UNRECOGNIZED_LIBRARY_MSG = 'Could not determine the model\'s numerical computation library of origin. ' \
                           + 'The library or library version is unsupported, or the model and its ' \
                           + 'parameters were passed incorrectly to loss-landscapes.' \
                           + 'Supported libraries include ' + str(SUPPORTED_LIBRARIES) \
                           + '. Check the documentation for how to correctly pass your ' \
                           + 'model and/or your parameter vectors to loss-landscapes.'


def determine_library(*args):
    """
    Returns a string flag representing the numerical computation library from which the
    model and the parameters have originated. Returns None if undetermined.
    """
    if isinstance(args[0], torch.nn.Module):
        for model in args[1:]:
            if not isinstance(model, torch.nn.Module):
                raise ValueError(UNRECOGNIZED_LIBRARY_MSG)
        return 'torch'
    else:
        raise ValueError(UNRECOGNIZED_LIBRARY_MSG)


def _is_torch_state(parameters) -> bool:
    """ Returns true if the given parameters are a list of torch.nn.Parameter objects. """
    if not isinstance(parameters, list):
        return False
    for p in parameters:
        if not isinstance(p, torch.nn.Parameter):
            return False
    return True
