"""
The main functionality of the loss_landscapes package is defined in this module.

In general, the functions in this module require the user to provide a neural network
model, as well as a function for evaluating the model's loss according to certain criteria,
and returns an array representing the model's loss landscape as seen in a part of some
subspace of the model's parameter space.

The available functions detect which numerical computation library is being used.
For example, if using PyTorch, pass your torch.nn.Module model where a model is
required.
"""


import copy
import numpy as np
from .utils.formats import determine_library
import loss_landscapes.backends.torch.compute
import loss_landscapes.backends.torch.ops


NOT_SUPPORTED_MSG = 'The model state provided is from a numerical computation library that is not supported.'
SUPPORTED_NORMS = ['filter', 'layer', 'model', None]
SUPPORTED_LIBRARIES = ['pytorch']


def random_line(start_model, evaluation_f, distance=1, steps=100, normalization=None) -> np.ndarray:
    """
    Returns an approximation of the loss of the model along a linear subspace of the
    parameter space defined by a start point and a randomly sampled direction.

    That is, given a neural network model, whose parameters define a point in parameter
    space, and a distance, the loss is computed at 'steps' points along a random
    direction, from the start point up to the maximum distance from the start point.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled vector in parameter space. That is, the more parameters
    a model has, the longer the distance in the random vector's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction vector according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction vector.

    Note also that a simple line approximation can produce misleading views
    of the loss landscape due to the scale invariance of neural networks. The sharpness or
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    normalize the direction, preferably with the 'filter' option.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.

    Args:
        start_model: model to be evaluated, whose current parameters represent the start point
        evaluation_f: function of form evaluation_f(model), used to evaluate model loss
        distance: maximum distance in parameter space from the start point
        steps: at how many steps from start to end the model is evaluated
        normalization: normalization of direction vector, must be one of 'filter', 'layer', 'model'

    Returns:
        np.ndarray: 1-d array of loss values along the randomly sampled direction
    """
    # determine which library the user has,
    library = determine_library(start_model)

    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(start_model)
    model_parameters = _get_parameters(model_copy, library)

    # get random direction
    direction = _sample_uniform_like(model_parameters, library, unit_vector=True)

    # normalize direction if required
    if normalization not in SUPPORTED_NORMS:
        raise ValueError('Invalid normalization method. Supported: ' + str(SUPPORTED_NORMS))
    elif normalization is not None:
        direction = _get_advanced_normalized_vector(direction, model_parameters, normalization, library)

    # compute and return losses
    return _compute_line(model_copy, direction, distance, steps, evaluation_f, library)


def linear_interpolation(model_start, model_end, evaluation_f, steps=100) -> np.ndarray:
    """
    Returns an approximation of the loss of the model along a linear subspace of the
    parameter space defined by two end points.

    That is, given two models, for both of which the model's parameters define a
    vertex in parameter space, the loss is computed at the given number of steps
    along the straight line connecting the two vertices. A common choice is to
    use the weights before training and the weights after convergence as the start
    and end points of the line, thus obtaining a view of the "straight line" in
    paramater space from the initialization to some minima. There is no guarantee
    that the model followed this path during optimization. In fact, it is highly
    unlikely to have done so, unless the optimization problem is convex.

    Note that a simple linear interpolation can produce misleading approximations
    of the loss landscape due to the scale invariance of neural networks. The sharpness/
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    use random_line() with filter normalization instead.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.

    Args:
        model_start: the model defining the start point of the line in parameter space
        model_end: the model defining the end point of the line in parameter space
        evaluation_f: function of form evaluation_f(model), used to evaluate model loss
        steps: at how many steps from start to end the model is evaluated

    Returns:
        np.ndarray: 1-d array of loss values along the line connecting start and end models
    """
    library = determine_library(model_start, model_end)

    # avoid aliasing issues by working on fresh copies
    model_start_copy = copy.deepcopy(model_start)
    model_end_copy = copy.deepcopy(model_end)
    model_start_parameters = _get_parameters(model_start_copy, library)
    model_end_parameters = _get_parameters(model_end_copy, library)

    # get distance and direction from start point to end point
    distance = _get_l2_norm(_get_displacement(model_start_parameters, model_end_parameters, library), library)
    direction = _get_unit_vector(_get_displacement(model_start_parameters, model_end_parameters, library), library)

    # compute and return losses
    return _compute_line(model_start_copy, direction, distance, steps, evaluation_f, library)


def random_plane(model_start, evaluation_f, distance=1, steps=100, normalization=None) -> np.ndarray:
    """
    Returns an approximation of the loss of the model along a planar subspace of the
    parameter space defined by a start point and two randomly sampled directions.

    That is, given a neural network model, whose parameters define a point in parameter
    space, and a distance, the loss is computed at 'steps' * 'steps' points along the
    plane defined by the two random directions, from the start point up to the maximum
    distance in both directions.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled vector in parameter space. That is, the more parameters
    a model has, the longer the distance in the random vector's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction vector according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction vector.

    Note also that a simple planar approximation with randomly sampled directions can produce
    misleading approximations of the loss landscape due to the scale invariance of neural
    networks. The sharpness/flatness of minima or maxima is affected by the scale of the neural
    network weights. For more details, see `https://arxiv.org/abs/1712.09913v3`. It is
    recommended to normalize the directions, preferably with the 'filter' option.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.

    Args:
        model_start: the model defining the origin point of the plane in parameter space
        evaluation_f: function of form evaluation_f(model), used to evaluate model loss
        distance: maximum distance in parameter space from the start point
        steps: at how many steps from start to end the model is evaluated
        normalization: normalization of direction vectors, must be one of 'filter', 'layer', 'model'

    Returns:
        np.ndarray: 1-d array of loss values along the line connecting start and end models
    """
    library = determine_library(model_start)

    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(model_start)
    model_parameters = _get_parameters(model_copy, library)

    direction_one = _sample_uniform_like(model_parameters, library, unit_vector=True)
    direction_two = _sample_uniform_like(model_parameters, library, unit_vector=True)

    # normalize if required
    if normalization not in SUPPORTED_NORMS:
        raise ValueError('Invalid normalization method. Supported: ' + str(SUPPORTED_NORMS))
    elif normalization is not None:
        direction_one = _get_advanced_normalized_vector(direction_one, model_parameters, normalization, library)
        direction_two = _get_advanced_normalized_vector(direction_two, model_parameters, normalization, library)

    # compute and return losses
    return _compute_plane(model_copy, direction_one, direction_two, distance, distance, steps, evaluation_f, library)


def planar_interpolation(model_start, model_end_one, model_end_two, evaluation_f, steps=100) -> np.ndarray:
    """
    Returns an approximation of the loss of the model along a planar subspace of the
    parameter space defined by a start point and two end points.

        That is, given two models, for both of which the model's parameters define a
    vertex in parameter space, the loss is computed at the given number of steps
    along the straight line connecting the two vertices. A common choice is to
    use the weights before training and the weights after convergence as the start
    and end points of the line, thus obtaining a view of the "straight line" in
    paramater space from the initialization to some minima. There is no guarantee
    that the model followed this path during optimization. In fact, it is highly
    unlikely to have done so, unless the optimization problem is convex.

    That is, given three neural network models, 'model_start', 'model_end_one', and
    'model_end_two', each of which defines a point in parameter space, the loss is
    computed at 'steps' * 'steps' points along the plane defined by the start vertex
    and the two vectors (end_one - start) and (end_two - start), up to the maximum
    distance in both directions. A common choice would be for two of the points to be
    the model after initialization, and the model after convergence. The third point
    could be another randomly initialized model, since in a high-dimensional space
    randomly sampled directions are most likely to be orthogonal.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.

    Args:
        model_start: the model defining the origin point of the plane in parameter space
        model_end_one: the model representing the end point of the first direction defining the plane
        model_end_two: the model representing the end point of the second direction defining the plane
        evaluation_f: function of form evaluation_f(model), used to evaluate model loss
        steps: at how many steps from start to end the model is evaluated

    Returns:
        np.ndarray: 1-d array of loss values along the line connecting start and end models
    """
    library = determine_library(model_start, model_end_two, model_end_two)

    # avoid aliasing issues by working on fresh copies
    model_start_copy = copy.deepcopy(model_start)
    model_end_one_copy = copy.deepcopy(model_end_one)
    model_end_two_copy = copy.deepcopy(model_end_two)
    start_parameters = _get_parameters(model_start_copy, library)
    end_one_parameters = _get_parameters(model_end_one_copy, library)
    end_two_parameters = _get_parameters(model_end_two_copy, library)

    # get numpy start point and direction
    direction_one = _get_displacement(start_parameters, end_one_parameters, library)
    direction_two = _get_displacement(start_parameters, end_two_parameters, library)
    distance_one = _get_l2_norm(direction_one, library)
    distance_two = _get_l2_norm(direction_two, library)

    # compute and return losses
    return _compute_plane(model_start_copy, direction_one, direction_two, distance_one, distance_two, steps,
                          evaluation_f, library, False)


def _compute_line(model, direction, distance, steps, evaluation_f, library):
    # dispatches computation along a line to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.compute.line(model, direction, distance, steps, evaluation_f)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _compute_line')


def _compute_plane(model, direction_one, direction_two, distance_one, distance_two, steps, evaluation_f, library,
                   center=False):
    # dispatches computation along a plane to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.compute.plane(model, direction_one, direction_two, distance_one,
                                                            distance_two, steps,
                                                            evaluation_f, center)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _compute_plane')


def _get_parameters(model, library):
    # dispatches model parameter extraction to the appropriate backend
    if library == 'torch':
        return list(model.parameters())
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_parameters')


def _sample_uniform_like(vector, library, unit_vector=True):
    # dispatches random sampling of a vector from uniform distribution to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.compute.sample_uniform_like(vector, unit_vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _sample_uniform_like')


def _get_advanced_normalized_vector(direction, model_parameters, norm_type, library):
    # dispatches application of filter normalization (and other norms) to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.compute.get_normalized_vector(direction, model_parameters, norm_type)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_normalized_vector')


def _get_unit_vector(vector, library):
    # dispatches computing the unit vector of a direction to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.unit_vector(vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_unit_vector')


def _get_displacement(point_a, point_b, library):
    # dispatches computing the displacement between two points to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.vector_subtraction(point_b, point_a)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_displacement')


def _get_l2_norm(vector, library):
    # dispatches computing the length of a vector to the appropriate backend
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.l2_norm(vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_l2_norm')
