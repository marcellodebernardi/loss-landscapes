"""
Functions for approximating loss/return landscapes in one and two dimensions.
"""

import copy
import typing
import torch.nn
import numpy as np
from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import rand_u_like, orthogonal_to
from loss_landscapes.metrics.metric import Metric


# noinspection DuplicatedCode
def point(model: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric) -> tuple:
    """
    Returns the computed value of the evaluation function applied to the model
    or agent at a specific point in parameter space.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric
    class, and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    The model supplied can be either a torch.nn.Module model, or a ModelWrapper from the
    loss_landscapes library for more complex cases.

    :param model: the model or model wrapper defining the point in parameter space
    :param metric: Metric object used to evaluate model
    :return: quantity specified by Metric at point in parameter space
    """
    return metric(wrap_model(model))


# noinspection DuplicatedCode
def linear_interpolation(model_start: typing.Union[torch.nn.Module, ModelWrapper],
                         model_end: typing.Union[torch.nn.Module, ModelWrapper],
                         metric: Metric, steps=100, deepcopy_model=False) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or
    agent along a linear subspace of the parameter space defined by two end points.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

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

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: the model defining the start point of the line in parameter space
    :param model_end: the model defining the end point of the line in parameter space
    :param metric: list of function of form evaluation_f(model), used to evaluate model loss
    :param steps: at how many steps from start to end the model is evaluated
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :return: 1-d array of loss values along the line connecting start and end models
    """
    # create wrappers from deep copies to avoid aliasing if desired
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    end_model_wrapper = wrap_model(copy.deepcopy(model_end) if deepcopy_model else model_end)

    start_point = model_start_wrapper.get_module_parameters()
    end_point = end_model_wrapper.get_module_parameters()
    direction = (end_point - start_point) / steps

    data_values = []
    for i in range(steps):
        # add a step along the line to the model parameters, then evaluate
        start_point.add_(direction)
        data_values.append(metric(model_start_wrapper))

    return np.array(data_values)


# noinspection DuplicatedCode
def random_line(model_start: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric, distance=0.1, steps=100,
                normalization='filter', deepcopy_model=False) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or agent along a
    linear subspace of the parameter space defined by a start point and a randomly sampled direction.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

    That is, given a neural network model, whose parameters define a point in parameter
    space, and a distance, the evaluation is computed at 'steps' points along a random
    direction, from the start point up to the maximum distance from the start point.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled other in parameter space. That is, the more parameters
    a model has, the longer the distance in the random other's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction other according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction other.

    Note also that a simple line approximation can produce misleading views
    of the loss landscape due to the scale invariance of neural networks. The sharpness or
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    normalize the direction, preferably with the 'filter' option.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: model to be evaluated, whose current parameters represent the start point
    :param metric: function of form evaluation_f(model), used to evaluate model loss
    :param distance: maximum distance in parameter space from the start point
    :param steps: at how many steps from start to end the model is evaluated
    :param normalization: normalization of direction other, must be one of 'filter', 'layer', 'model'
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :return: 1-d array of loss values along the randomly sampled direction
    """
    # create wrappers from deep copies to avoid aliasing if desired
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)

    # obtain start point in parameter space and random direction
    # random direction is randomly sampled, then normalized, and finally scaled by distance/steps
    start_point = model_start_wrapper.get_module_parameters()
    direction = rand_u_like(start_point)

    if normalization == 'model':
        direction.model_normalize_(start_point)
    elif normalization == 'layer':
        direction.layer_normalize_(start_point)
    elif normalization == 'filter':
        direction.filter_normalize_(start_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

    direction.mul_(((start_point.model_norm() * distance) / steps) / direction.model_norm())

    data_values = []
    for i in range(steps):
        # add a step along the line to the model parameters, then evaluate
        start_point.add_(direction)
        data_values.append(metric(model_start_wrapper))

    return np.array(data_values)


# noinspection DuplicatedCode
def planar_interpolation(model_start: typing.Union[torch.nn.Module, ModelWrapper],
                         model_end_one: typing.Union[torch.nn.Module, ModelWrapper],
                         model_end_two: typing.Union[torch.nn.Module, ModelWrapper],
                         metric: Metric, steps=20, deepcopy_model=False) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or agent along
    a planar subspace of the parameter space defined by a start point and two end points.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

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

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model_start: the model defining the origin point of the plane in parameter space
    :param model_end_one: the model representing the end point of the first direction defining the plane
    :param model_end_two: the model representing the end point of the second direction defining the plane
    :param metric: function of form evaluation_f(model), used to evaluate model loss
    :param steps: at how many steps from start to end the model is evaluated
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :return: 1-d array of loss values along the line connecting start and end models
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    model_end_one_wrapper = wrap_model(copy.deepcopy(model_end_one) if deepcopy_model else model_end_one)
    model_end_two_wrapper = wrap_model(copy.deepcopy(model_end_two) if deepcopy_model else model_end_two)

    # compute direction vectors
    start_point = model_start_wrapper.get_module_parameters()
    dir_one = (model_end_one_wrapper.get_module_parameters() - start_point) / steps
    dir_two = (model_end_two_wrapper.get_module_parameters() - start_point) / steps

    data_matrix = []
    # evaluate loss in grid of (steps * steps) points, where each column signifies one step
    # along dir_one and each row signifies one step along dir_two. The implementation is again
    # a little convoluted to avoid constructive operations. Fundamentally we generate the matrix
    # [[start_point + (dir_one * i) + (dir_two * j) for j in range(steps)] for i in range(steps].
    for i in range(steps):
        data_column = []

        for j in range(steps):
            # for every other column, reverse the order in which the column is generated
            # so you can easily use in-place operations to move along dir_two
            if i % 2 == 0:
                start_point.add_(dir_two)
                data_column.append(metric(model_start_wrapper))
            else:
                start_point.sub_(dir_two)
                data_column.insert(0, metric(model_start_wrapper))

        data_matrix.append(data_column)
        start_point.add_(dir_one)

    return np.array(data_matrix)


# noinspection DuplicatedCode
def random_plane(model: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric, distance=1, steps=20,
                 normalization='filter', deepcopy_model=False) -> np.ndarray:
    """
    Returns the computed value of the evaluation function applied to the model or agent along a planar
    subspace of the parameter space defined by a start point and two randomly sampled directions.
    The models supplied can be either torch.nn.Module models, or ModelWrapper objects
    from the loss_landscapes library for more complex cases.

    That is, given a neural network model, whose parameters define a point in parameter
    space, and a distance, the loss is computed at 'steps' * 'steps' points along the
    plane defined by the two random directions, from the start point up to the maximum
    distance in both directions.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled other in parameter space. That is, the more parameters
    a model has, the longer the distance in the random other's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction other according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction other.

    Note also that a simple planar approximation with randomly sampled directions can produce
    misleading approximations of the loss landscape due to the scale invariance of neural
    networks. The sharpness/flatness of minima or maxima is affected by the scale of the neural
    network weights. For more details, see `https://arxiv.org/abs/1712.09913v3`. It is
    recommended to normalize the directions, preferably with the 'filter' option.

    The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

    :param model: the model defining the origin point of the plane in parameter space
    :param metric: function of form evaluation_f(model), used to evaluate model loss
    :param distance: maximum distance in parameter space from the start point
    :param steps: at how many steps from start to end the model is evaluated
    :param normalization: normalization of direction vectors, must be one of 'filter', 'layer', 'model'
    :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
    :return: 1-d array of loss values along the line connecting start and end models
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model) if deepcopy_model else model)

    start_point = model_start_wrapper.get_module_parameters()
    dir_one = rand_u_like(start_point)
    dir_two = orthogonal_to(dir_one)

    if normalization == 'model':
        dir_one.model_normalize_(start_point)
        dir_two.model_normalize_(start_point)
    elif normalization == 'layer':
        dir_one.layer_normalize_(start_point)
        dir_two.layer_normalize_(start_point)
    elif normalization == 'filter':
        dir_one.filter_normalize_(start_point)
        dir_two.filter_normalize_(start_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')

    # scale to match steps and total distance
    dir_one.mul_(((start_point.model_norm() * distance) / steps) / dir_one.model_norm())
    dir_two.mul_(((start_point.model_norm() * distance) / steps) / dir_two.model_norm())
    # Move start point so that original start params will be in the center of the plot
    dir_one.mul_(steps / 2)
    dir_two.mul_(steps / 2)
    start_point.sub_(dir_one)
    start_point.sub_(dir_two)
    dir_one.truediv_(steps / 2)
    dir_two.truediv_(steps / 2)

    data_matrix = []
    # evaluate loss in grid of (steps * steps) points, where each column signifies one step
    # along dir_one and each row signifies one step along dir_two. The implementation is again
    # a little convoluted to avoid constructive operations. Fundamentally we generate the matrix
    # [[start_point + (dir_one * i) + (dir_two * j) for j in range(steps)] for i in range(steps].
    for i in range(steps):
        data_column = []

        for j in range(steps):
            # for every other column, reverse the order in which the column is generated
            # so you can easily use in-place operations to move along dir_two
            if i % 2 == 0:
                start_point.add_(dir_two)
                data_column.append(metric(model_start_wrapper))
            else:
                start_point.sub_(dir_two)
                data_column.insert(0, metric(model_start_wrapper))

        data_matrix.append(data_column)
        start_point.add_(dir_one)

    return np.array(data_matrix)


# todo add hypersphere function
