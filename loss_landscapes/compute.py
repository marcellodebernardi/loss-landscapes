import copy
import numpy as np
import linalg.vectors as vec
from externals.model_converters import build_converter
from state_representation import LayeredVector

# import tensorflow as tf
# import tensorflow.keras


NOT_SUPPORTED_MSG = 'The model state provided is from a numerical computation library that is not supported.'
SUPPORTED_NORMS = ['filter', 'layer', 'model', None]


def random_line(model, start_p, evaluation_f, distance=1, steps=100, normalization=None):
    """
    Returns an approximation of the loss of the model along a linear subspace of the
    parameter space defined by a start point and a randomly sampled direction.

    That is, given a set of parameters 'start_p', which defines a point in parameter
    space, and a distance, the loss is computed at 'steps' points along the random
    direction, from the start point up to the maximum distance from the start point.

    Note that a simple line approximation can produce misleading approximations
    of the loss landscape due to the scale invariance of neural networks. The sharpness/
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    normalize the direction, preferably with the 'filter' option.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.
    """
    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(model)
    start_p_copy = copy.deepcopy(start_p)

    # get numpy start point and direction
    model_converter = build_converter(model_copy)
    start_p_copy = model_converter.to_internal(start_p_copy)
    direction = vec.unit_vector(_sample_uniform_like(start_p_copy))

    # normalize if required
    if normalization not in SUPPORTED_NORMS:
        raise ValueError('Invalid normalization method. Supported: ' + str(SUPPORTED_NORMS))
    elif normalization is not None:
        direction = _normalize(direction, start_p_copy, normalization)

    # compute and return losses
    return _compute_line(model_converter, direction, distance, steps, evaluation_f)


def linear_interpolation(model, start_p, end_p, evaluation_f, steps=100):
    """
    Returns an approximation of the loss of the model along a linear subspace of the
    parameter space defined by two end points.

    That is, given a set of parameters 'start_p' and 'end_p', both of which
    define a point in parameter space, the loss is computed at 'steps' points
    along the straight line connecting the two points. A common choice is to
    use the weights before training and the weights after convergence as the start
    and end points of the line.

    Note that a simple linear interpolation can produce misleading approximations
    of the loss landscape due to the scale invariance of neural networks. The sharpness/
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    use random_line() with filter normalization instead.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.
    """
    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(model)
    start_p_copy = copy.deepcopy(start_p)
    end_p_copy = copy.deepcopy(end_p)

    # get numpy start point and direction
    model_converter = build_converter(model_copy)
    start_p_copy = model_converter.to_internal(start_p_copy)
    end_p_copy = model_converter.to_internal(end_p_copy)
    distance = vec.l2_norm(vec.subtraction(end_p_copy, start_p_copy))
    direction = vec.unit_vector(vec.subtraction(end_p_copy, start_p_copy))

    # compute and return losses
    return _compute_line(model_converter, direction, distance, steps, evaluation_f)


def random_plane(model, start_p, evaluation_f, distance=1, steps=100, normalization=None, center=True):
    """
    Returns an approximation of the loss of the model along a planar subspace of the
    parameter space defined by a start point and two randomly sampled directions.

    That is, given a set of parameters 'start_p', which defines a point in parameter
    space, and a distance, the loss is computed at 'steps' * 'steps' points along the
    plane defined by the two random directions, from the start point up to the maximum
    distance from the start point in both directions.

    Note that a simple planar approximation with randomly sampled directions can produce
    misleading approximations of the loss landscape due to the scale invariance of neural
    networks. The sharpness/flatness of minima or maxima is affected by the scale of the neural
    network weights. For more details, see `https://arxiv.org/abs/1712.09913v3`. It is
    recommended to normalize the directions, preferably with the 'filter' option.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.
    """
    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(model)
    start_p_copy = copy.deepcopy(start_p)

    # get numpy start point and directions - note that in very high dimensional
    # space two uniformly sampled directions are very likely to be nearly orthogonal
    model_converter = build_converter(model_copy)
    start_p_copy = model_converter.to_internal(start_p_copy)
    direction_one = vec.unit_vector(_sample_uniform_like(start_p_copy))
    direction_two = vec.unit_vector(_sample_uniform_like(start_p_copy))

    # normalize if required
    if normalization not in SUPPORTED_NORMS:
        raise ValueError('Invalid normalization method. Supported: ' + str(SUPPORTED_NORMS))
    elif normalization is not None:
        direction_one = _normalize(direction_one, start_p_copy, normalization)
        direction_two = _normalize(direction_two, start_p_copy, normalization)

    # compute and return losses
    return _compute_plane(model_converter, direction_one, direction_two, distance, steps, evaluation_f, center)


def planar_interpolation(model, start_p, end_p_one, end_p_two, evaluation_f, steps=100):
    """
    Returns an approximation of the loss of the model along a planar subspace of the
    parameter space defined by a start point and two end points.

    That is, given sets of parameters 'start_p', 'end_p_one', and 'end_p_two', all
    of which define a point in parameter space, the loss is computed at 'steps' * 'steps'
    points along the plane defined by the two directions from the start point to the end
    points, up to the maximum distance in both directions.

    Note that a simple planar interpolation can produce misleading approximations
    of the loss landscape due to the scale invariance of neural networks. The sharpness/
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to use
    random_plane() with filter normalization.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.
    """
    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(model)
    start_p_copy = copy.deepcopy(start_p)
    end_p_one_copy = copy.deepcopy(end_p_one)
    end_p_two_copy = copy.deepcopy(end_p_two)

    # get numpy start point and direction
    model_converter = build_converter(model_copy)
    start_p_copy = model_converter.to_internal(start_p_copy)
    end_p_one_copy = model_converter.to_internal(end_p_one_copy)
    end_p_two_copy = model_converter.to_internal(end_p_two_copy)
    distance = vec.l2_norm(vec.subtraction(end_p_one_copy, start_p_copy))
    direction_one = vec.unit_vector(vec.subtraction(end_p_one_copy, start_p_copy))
    direction_two = vec.unit_vector(vec.subtraction(end_p_two_copy, start_p_copy))

    # compute and return losses
    return _compute_plane(model_converter, direction_one, direction_two, distance, steps, evaluation_f, False)


def _compute_line(model_converter, direction, distance, steps, evaluation) -> np.ndarray:
    """
    Computes loss for discretized intervals on an one-dimensional parameter line.

    Args:
        model_converter: converter that wraps the model and updates its parameters
        direction: direction vector in parameter space
        distance: distance along direction to reach
        steps: how many steps along direction to reach distance
        evaluation: model evaluation function
    """
    losses = []
    direction_step = vec.scalar_multiplication(direction, distance / steps)

    for _ in range(steps):
        model_converter.add(direction_step)
        losses.append(evaluation(model_converter.model()))

    return np.array(losses)


def _compute_plane(model_converter, direction_one, direction_two, distance, steps, evaluation, center) -> np.ndarray:
    """
    Computes loss for discretized intervals on a two-dimensional parameter plane.

    Args:
        model_converter: converter that wraps the model and updates its parameters
        direction_one: first direction vector in parameter space
        direction_two: second direction vector in parameter space
        distance: distance along direction to reach
        steps: how many steps along direction to reach distance
        evaluation: model evaluation function
        center: whether the start point should be the top-left or central point of the plane
    """
    losses = []
    direction_one_step = vec.scalar_multiplication(direction_one, distance / steps)
    direction_two_step = vec.scalar_multiplication(direction_two, distance / steps)
    total_distance_two = vec.scalar_multiplication(direction_two, distance)

    for _ in range(steps):
        inner_losses = []
        for _ in range(steps):
            model_converter.add(direction_two_step)
            inner_losses.append(evaluation(model_converter.model()))

        losses.append(inner_losses)
        model_converter.subtract(total_distance_two)
        model_converter.add(direction_one_step)

    return np.array(losses)


def _normalize(direction, model_parameters, normalization):
    if normalization == 'filter':
        return _apply_filter_norm(direction, model_parameters)
    elif normalization == 'layer':
        return _apply_layer_norm(direction, model_parameters)
    elif normalization == 'model':
        return _apply_model_norm(direction, model_parameters)


def _apply_filter_norm(direction: LayeredVector, model_parameters: LayeredVector) -> LayeredVector:
    """ Applies filter normalization to a direction vector """
    new_vector = LayeredVector()

    # iterate over every layer's every filter
    for layer_idx in range(len(model_parameters)):
        new_layer = []
        for filter_idx in range(model_parameters[layer_idx].shape(0)):
            # for each filter, the corresponding new filter is obtained by the equation in the paper
            direction_filter = direction[layer_idx, filter_idx]
            frob_norm_d = np.linalg.norm(direction_filter, ord='fro')
            frob_norm_p = np.linalg.norm(model_parameters[layer_idx, filter_idx], ord='fro')

            new_filter = np.multiply(np.division(direction_filter, frob_norm_d), frob_norm_p)
            new_layer.append(np.ndarray(new_filter))

        new_vector.add_layer(np.array(new_layer))

    return new_vector


def _apply_layer_norm(direction: LayeredVector, model_parameters: LayeredVector) -> LayeredVector:
    """ Applies layer normalization to a direction vector """
    # todo
    return direction


def _apply_model_norm(direction: LayeredVector, model_parameters: LayeredVector) -> LayeredVector:
    """ Applies model normalization to a direction vector """
    # todo
    return direction


def _sample_uniform_like(parameters: LayeredVector) -> LayeredVector:
    """ Returns a direction vector compatible with the given parameters sample from uniform distribution. """
    direction = LayeredVector()

    for p in parameters:
        direction.add_layer(np.random.uniform(-1, 1, np.shape(p)))

    return direction
