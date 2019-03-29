import numpy as np
import loss_landscapes.compute


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
    return loss_landscapes.compute.compute_random_line(start_model, evaluation_f, distance, steps, normalization)


def linear_interpolation(model_start, model_end, evaluation_f, steps=100):
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
    return loss_landscapes.compute.compute_linear_interpolation(model_start, model_end, evaluation_f, steps)


def random_plane(model_start, evaluation_f, distance=1, steps=100, normalization=None, center=False) -> np.ndarray:
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
        center: whether the start point is used as the central point or the start point

    Returns:
        np.ndarray: 1-d array of loss values along the line connecting start and end models
    """
    return loss_landscapes.compute.compute_random_plane(model_start, evaluation_f, distance, steps, normalization,
                                                        center, None, None)


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
    return loss_landscapes.compute.compute_planar_interpolation(model_start, model_end_one, model_end_two, evaluation_f,
                                                                steps, None, None)
