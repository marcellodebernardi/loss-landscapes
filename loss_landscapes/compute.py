import copy
from .utils.formats import determine_library
import loss_landscapes.backends.torch.compute
import loss_landscapes.backends.torch.ops

NOT_SUPPORTED_MSG = 'The model state provided is from a numerical computation library that is not supported.'
SUPPORTED_NORMS = ['filter', 'layer', 'model', None]
SUPPORTED_LIBRARIES = ['pytorch']


def random_line(model, evaluation_f, distance=1, steps=100, normalization=None):
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
    # determine which library the user has,
    library = determine_library(model)

    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(model)
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


def linear_interpolation(model_start, model_end, evaluation_f, steps=100):
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


def random_plane(model, evaluation_f, distance=1, steps=100, normalization=None, center=False):
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
    library = determine_library(model)

    # avoid aliasing issues by working on fresh copies
    model_copy = copy.deepcopy(model)
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
    return _compute_plane(model_copy, direction_one, direction_two, distance, distance, steps, evaluation_f, library,
                          center)


def planar_interpolation(model_start, model_end_one, model_end_two, evaluation_f, steps=100):
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
    if library == 'torch':
        return loss_landscapes.backends.torch.compute.line(model, direction, distance, steps, evaluation_f)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _compute_line')


def _compute_plane(model, direction_one, direction_two, distance_one, distance_two, steps, evaluation_f, library,
                   center=False):
    if library == 'torch':
        return loss_landscapes.backends.torch.compute.plane(model, direction_one, direction_two, distance_one,
                                                            distance_two, steps,
                                                            evaluation_f, center)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _compute_plane')


def _get_parameters(model, library):
    if library == 'torch':
        return list(model.parameters())
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_parameters')


def _sample_uniform_like(vector, library, unit_vector=True):
    if library == 'torch':
        return loss_landscapes.backends.torch.compute.sample_uniform_like(vector, unit_vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _sample_uniform_like')


def _get_advanced_normalized_vector(direction, model_parameters, norm_type, library):
    if library == 'torch':
        return loss_landscapes.backends.torch.compute.get_normalized_vector(direction, model_parameters, norm_type)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_normalized_vector')


def _get_unit_vector(vector, library):
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.unit_vector(vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_unit_vector')


def _get_displacement(point_a, point_b, library):
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.vector_subtraction(point_b, point_a)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_displacement')


def _get_l2_norm(vector, library):
    if library == 'torch':
        return loss_landscapes.backends.torch.ops.l2_norm(vector)
    else:
        raise ValueError('Invalid library flag ' + str(library) + ' passed to _get_l2_norm')
