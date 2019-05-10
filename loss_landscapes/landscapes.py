"""
Functions for approximating landscapes in one and two dimensions.
"""

import copy
import loss_landscapes.model_interface.parameter_vector as pv
from loss_landscapes.model_interface.model_wrapper import ModelWrapper
from loss_landscapes.evaluators.evaluator import Evaluator


def point(model, evaluator: Evaluator) -> tuple:
    """
    Returns the computed value of the evaluation function applied to the model
    at a specific point in parameter space.

    The Evaluator supplied has to be a subclass of the evaluations.evaluator.Evaluator class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model: the model defining the point in parameter space
    :param evaluator: list of function of form evaluation_f(model), used to evaluate model loss
    :return: quantity specified by evaluation_f at point in parameter space
    """
    return evaluator(model)


def linear_interpolation(model_start, model_end, evaluator: Evaluator, steps=100) -> list:
    """
    Returns the computed value of the evaluation function applied to the model 
    along a linear subspace of the parameter space defined by two end points.

    That is, given two models, for both of which the model's parameters define a
    vertex in parameter space, the evaluation is computed at the given number of steps
    along the straight line connecting the two vertices. A common choice is to
    use the weights before training and the weights after convergence as the start
    and end points of the line, thus obtaining a view of the "straight line" in
    parameter space from the initialization to some minima. There is no guarantee
    that the model followed this path during optimization. In fact, it is highly
    unlikely to have done so, unless the optimization problem is convex.

    Note that a simple linear interpolation can produce misleading approximations
    of the loss landscape due to the scale invariance of neural networks. The sharpness/
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    use random_line() with filter normalization instead.

    The Evaluator supplied has to be a subclass of the evaluations.evaluator.Evaluator class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: the model defining the start point of the line in parameter space
    :param model_end: the model defining the end point of the line in parameter space
    :param evaluator: list of function of form evaluation_f(model), used to evaluate model loss
    :param steps: at how many steps from start to end the model is evaluated
    :return: 1-d array of loss values along the line connecting start and end models
    """
    # create wrappers from deep copies to avoid aliasing
    start_model_wrapper = ModelWrapper(copy.deepcopy(model_start))
    end_model_wrapper = ModelWrapper(copy.deepcopy(model_end))

    start_point = start_model_wrapper.build_parameter_vector()
    end_point = end_model_wrapper.build_parameter_vector()
    direction = (end_point - start_point) / steps

    data_values = []
    for i in range(steps):
        # add a step along the line to the model parameters, then evaluate
        start_model_wrapper.set_parameters(start_point + (direction * i))
        data_values.append(evaluator(start_model_wrapper.get_model()))

    return data_values


def random_line(model_start, evaluator: Evaluator, distance=1, steps=100, normalization=None) -> list:
    """
    Returns the computed value of the evaluation function applied to the model along a 
    linear subspace of the parameter space defined by a start point and a randomly sampled direction.

    That is, given a neural network model, whose parameters define a point in parameter
    space, and a distance, the evaluation is computed at 'steps' points along a random
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

    The Evaluator supplied has to be a subclass of the evaluations.evaluator.Evaluator class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: model to be evaluated, whose current parameters represent the start point
    :param evaluator: function of form evaluation_f(model), used to evaluate model loss
    :param distance: maximum distance in parameter space from the start point
    :param steps: at how many steps from start to end the model is evaluated
    :param normalization: normalization of direction vector, must be one of 'filter', 'layer', 'model'
    :return: 1-d array of loss values along the randomly sampled direction
    """
    # create wrappers from deep copies to avoid aliasing
    model_start_wrapper = ModelWrapper(copy.deepcopy(model_start))

    # obtain start point in parameter space and random direction
    # random direction is randomly sampled, then normalized, and finally scaled by distance/steps
    start_point = model_start_wrapper.build_parameter_vector()
    direction = pv.rand_u_like(start_point)

    if normalization == 'model':
        direction.model_normalize_()
    elif normalization == 'layer':
        direction.layer_normalize_()
    elif normalization == 'filter':
        direction.filter_normalize_()
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

    direction.mul_(distance / steps)

    data_values = []
    for i in range(steps):
        # add a step along the line to the model parameters, then evaluate
        model_start_wrapper.set_parameters(start_point + (direction * i))
        data_values.append(evaluator(model_start_wrapper.get_model()))

    return data_values


def planar_interpolation(model_start, model_end_one, model_end_two, evaluator: Evaluator, steps=20) -> list:
    """
    Returns the computed value of the evaluation function applied to the model along
    a planar subspace of the parameter space defined by a start point and two end points.

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

    The Evaluator supplied has to be a subclass of the evaluations.evaluator.Evaluator class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: the model defining the origin point of the plane in parameter space
    :param model_end_one: the model representing the end point of the first direction defining the plane
    :param model_end_two: the model representing the end point of the second direction defining the plane
    :param evaluator: function of form evaluation_f(model), used to evaluate model loss
    :param steps: at how many steps from start to end the model is evaluated
    :return: 1-d array of loss values along the line connecting start and end models
    """
    model_start_wrapper = ModelWrapper(copy.deepcopy(model_start))
    model_end_one_wrapper = ModelWrapper(copy.deepcopy(model_end_one))
    model_end_two_wrapper = ModelWrapper(copy.deepcopy(model_end_two))

    # compute direction vectors
    start_point = model_start_wrapper.build_parameter_vector()
    end_point_one = model_end_one_wrapper.build_parameter_vector()
    end_point_two = model_end_two_wrapper.build_parameter_vector()
    dir_one = (end_point_one - start_point) / steps
    dir_two = (end_point_two - start_point) / steps

    data_matrix = []
    # for each increment in direction one, evaluate all steps in direction two
    for i in range(steps):
        data_column = []

        for j in range(steps):
            # set parameters and evaluate
            model_start_wrapper.set_parameters(start_point + (dir_two * j))
            data_column.append(evaluator(model_start_wrapper.get_model()))
            # increment parameters

        data_matrix.append(data_column)
        start_point.add_(dir_one)

    return data_matrix


def random_plane(model_start, evaluator: Evaluator, distance=1, steps=20, normalization=None, center=True) -> list:
    """
    Returns the computed value of the evaluation function applied to the model along a planar
    subspace of the parameter space defined by a start point and two randomly sampled directions.

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

    The Evaluator supplied has to be a subclass of the evaluations.evaluator.Evaluator class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: the model defining the origin point of the plane in parameter space
    :param evaluator: function of form evaluation_f(model), used to evaluate model loss
    :param distance: maximum distance in parameter space from the start point
    :param steps: at how many steps from start to end the model is evaluated
    :param normalization: normalization of direction vectors, must be one of 'filter', 'layer', 'model'
    :param center: whether the start point is used as the central point or the start point
    :return: 1-d array of loss values along the line connecting start and end models
    """
    model_start_wrapper = ModelWrapper(copy.deepcopy(model_start))

    start_point = model_start_wrapper.build_parameter_vector()
    dir_one = pv.rand_u_like(start_point)
    dir_two = pv.rand_u_like(start_point)

    if normalization == 'model':
        dir_one.model_normalize_()
        dir_two.model_normalize_()
    elif normalization == 'layer':
        dir_one.layer_normalize_()
        dir_two.layer_normalize_()
    elif normalization == 'filter':
        dir_one.filter_normalize_()
        dir_two.filter_normalize_()
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

    dir_one.mul_(distance / steps)
    dir_two.mul_(distance / steps)

    if center:
        # if center, move start point in opposite direction of dir_one and dir_two by half the total distance
        start_point.sub_(dir_one * (steps / 2))
        start_point.sub_(dir_two * (steps / 2))

    data_matrix = []
    # for each increment in direction one, evaluate all steps in direction two
    for i in range(steps):
        data_column = []

        for j in range(steps):
            model_start_wrapper.set_parameters(start_point + (dir_two * j))
            data_column.append(evaluator(model_start_wrapper.get_model()))

        data_matrix.append(data_column)
        # reset dir two, increment dir one
        start_point.add_(dir_one)

    return data_matrix
