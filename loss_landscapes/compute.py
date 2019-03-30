"""
INTERNAL - the main functionality of the loss_landscapes package is defined in this module.

In general, the functions in this module contain the core high-level logic of all the
algorithms available in this package. The code in this module is intended to be wrapped
by client-facing functions in other modules, and in turns relies on a 'backend_dispatcher'
to dynamically select the right backend code to carry out its basic operations.
"""


import copy
import numpy as np
import loss_landscapes.backends.backend_dispatch as backend_dispatch
import loss_landscapes.utils.formats
import loss_landscapes.backends.torch.backend
import loss_landscapes.backends.torch.ops


NOT_SUPPORTED_MSG = 'The model state provided is from a numerical computation library that is not supported.'
SUPPORTED_NORMS = ['filter', 'layer', 'model', None]
SUPPORTED_LIBRARIES = ['pytorch']
TORCH_BACKEND = loss_landscapes.backends.torch.backend.TorchBackend()


def compute_random_line(start_model, evaluation_f, distance=1, steps=100, normalization=None, agent=None,
                        model_set_f=None) -> np.ndarray:
    """ Compute a random line, either for just a model, or for an agent wrapping a model. """
    # determine which library the user has,
    library = loss_landscapes.utils.formats.determine_library(start_model)

    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(start_model)
    model_parameters = backend_dispatch.get_parameters(model_copy, library)

    # get random direction
    direction = backend_dispatch.sample_uniform_like(model_parameters, library, unit_vector=True)

    # normalize direction if required
    if normalization not in SUPPORTED_NORMS:
        raise ValueError('Invalid normalization method. Supported: ' + str(SUPPORTED_NORMS))
    elif normalization is not None:
        direction = backend_dispatch.get_advanced_normalized_vector(direction, model_parameters, normalization, library)

    # compute and return losses
    return backend_dispatch.compute_line(model_copy, direction, distance, steps, library, evaluation_f, agent,
                                         model_set_f)


def compute_linear_interpolation(model_start, model_end, evaluation_f, steps=100, agent=None,
                                 model_set_f=None) -> np.ndarray:
    """ Compute a linear interpolation, either for just a model, or for an agent wrapping a model. """
    library = loss_landscapes.utils.formats.determine_library(model_start, model_end)

    # avoid aliasing issues by working on fresh copies
    model_start_copy = copy.deepcopy(model_start)
    model_end_copy = copy.deepcopy(model_end)
    model_start_parameters = backend_dispatch.get_parameters(model_start_copy, library)
    model_end_parameters = backend_dispatch.get_parameters(model_end_copy, library)

    # get distance and direction from start point to end point
    distance = backend_dispatch.get_l2_norm(
        backend_dispatch.get_displacement(model_start_parameters, model_end_parameters, library),
        library)
    direction = backend_dispatch.get_unit_vector(
        backend_dispatch.get_displacement(model_start_parameters, model_end_parameters, library), library)

    # compute and return losses
    return backend_dispatch.compute_line(model_start_copy, direction, distance, steps, library, evaluation_f, agent,
                                         model_set_f)


def compute_random_plane(model_start, evaluation_f, distance=1, steps=100, normalization=None, center=False, agent=None,
                         model_set_f=None) -> np.ndarray:
    """ Compute a random plane, either for just a model, or for an agent wrapping a model. """
    # determine library in use
    library = loss_landscapes.utils.formats.determine_library(model_start)

    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(model_start)
    model_parameters = backend_dispatch.get_parameters(model_copy, library)

    direction_one = backend_dispatch.sample_uniform_like(model_parameters, library, unit_vector=True)
    direction_two = backend_dispatch.sample_uniform_like(model_parameters, library, unit_vector=True)

    # normalize if required
    if normalization not in SUPPORTED_NORMS:
        raise ValueError('Invalid normalization method. Supported: ' + str(SUPPORTED_NORMS))
    elif normalization is not None:
        direction_one = backend_dispatch.get_advanced_normalized_vector(direction_one, model_parameters, normalization,
                                                                        library)
        direction_two = backend_dispatch.get_advanced_normalized_vector(direction_two, model_parameters, normalization,
                                                                        library)

    # compute and return losses
    return backend_dispatch.compute_plane(model_copy, direction_one, direction_two, distance, distance, steps, library,
                                          evaluation_f, center, agent, model_set_f)


def compute_planar_interpolation(model_start, model_end_one, model_end_two, evaluation_f, steps=100, agent=None,
                                 model_set_f=None) -> np.ndarray:
    """ Compute a planar interpolation, either for just a model, or for an agent wrapping a model. """
    library = loss_landscapes.utils.formats.determine_library(model_start, model_end_two, model_end_two)

    # avoid aliasing issues by working on fresh copies
    model_start_copy = copy.deepcopy(model_start)
    model_end_one_copy = copy.deepcopy(model_end_one)
    model_end_two_copy = copy.deepcopy(model_end_two)
    start_parameters = backend_dispatch.get_parameters(model_start_copy, library)
    end_one_parameters = backend_dispatch.get_parameters(model_end_one_copy, library)
    end_two_parameters = backend_dispatch.get_parameters(model_end_two_copy, library)

    # get numpy start point and direction
    direction_one = backend_dispatch.get_displacement(start_parameters, end_one_parameters, library)
    direction_two = backend_dispatch.get_displacement(start_parameters, end_two_parameters, library)
    distance_one = backend_dispatch.get_l2_norm(direction_one, library)
    distance_two = backend_dispatch.get_l2_norm(direction_two, library)

    # compute and return losses
    return backend_dispatch.compute_plane(model_start_copy, direction_one, direction_two, distance_one, distance_two,
                                          steps, library, evaluation_f, False, agent, model_set_f)
