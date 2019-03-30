"""
Includes the same functionality of the nn module, but for reinforcement learning.

That is, the functions in this module aren't intended for computing the loss landscapes of
a neural network. They are aimed at solving the more general problem of computing the
'performance landscape' of a program or agent that has a parametric component written
in a library such as PyTorch. For example, one could plot a 'reward landscape' of a
reinforcement learning algorithm like DQN in a particular environment. The interface to
the functions in this module is therefore necessarily more complex than the functions in
the library's main component, since the library can make less assumptions about the structure
of the agent it is evaluating.

The available functions detect which numerical computation library is being used.
For example, if using PyTorch, pass your torch.nn.Module model where a model is
required.
"""


import numpy as np
import loss_landscapes.compute


def random_line(agent, model, model_set_f, evaluation_f, distance=1, steps=100, normalization=None) -> np.ndarray:
    """
    Returns an approximation of the return attained by the agent in its environment,
    along a linear subspace of the model's parameter space defined by a start point
    and a randomly sampled direction.

    That is, given an agent somehow containing a neural network model, whose parameters
    define a point in parameter space, and a distance, the expected return of the agent
    on its task is computed at 'steps' points along a random direction, in parameter space,
     from the start point up to the maximum distance from the start.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled vector in parameter space. That is, the more parameters
    a model has, the longer the distance in the random vector's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction vector according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction vector.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss. The model setting function has to
    be of the form

        model_set_function(agent, model)

    and must specify a procedure by which the agent's internal parametric model is set
    to be the one given as an argument.

    :param agent: agent to evaluate
    :param model: parametric model used in agent
    :param model_set_f: a function that defines how to set the agent's internal model
    :param evaluation_f: a function that defines how to compute the expected reward of an agent
    :param distance: maximum distance along the line in parameter space
    :param steps: number of steps to divide the distance into
    :param normalization: normalization method for the direction vector
    :return: 1-d array of return values along the line connecting start and end models
    """
    return loss_landscapes.compute.compute_random_line(model, evaluation_f, distance, steps, normalization, agent,
                                                       model_set_f)


def linear_interpolation(agent, model_start, model_end, model_set_f, evaluation_f, steps=100) -> np.ndarray:
    """
    Returns an approximation of the reward attained by the agent in its environment,
    along a linear subspace of the internal model's parameter space as defined by two
    end points.

    That is, given an agent, and two models, both of which define a particular state
    of the agent as well as a vertex in parameter space, the expected reward for the
    agent is computed at the given number of steps along the straight line connecting
    the two vertices. A common choice is to use the weights before training and the
    weights after convergence as the start and end points of the line, thus obtaining
    a view of the "straight line" in parameter space from the initialization to some
    minima. There is no guarantee that the model followed this path during optimization.
    In fact, it is highly unlikely to have done so, unless the optimization problem is convex.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss. The model setting function has to
    be of the form

        model_set_function(agent, model)

    and must specify a procedure by which the agent's internal parametric model is set
    to be the one given as an argument.

    :param agent: agent to evaluate
    :param model_start: parametric model used in agent, represents start point of linear interpolation
    :param model_end: parametric model used in agent, represents end point of linear interpolation
    :param model_set_f: a function that defines how to set the agent's internal model
    :param evaluation_f: a function that defines how to compute the expected reward of an agent
    :param steps: number of steps to divide the distance into
    :return: 1-d array of return values along the line connecting start and end models
    :return:
    """
    return loss_landscapes.compute.compute_linear_interpolation(model_start, model_end, evaluation_f, steps, agent,
                                                                model_set_f)


def random_plane(agent, model, model_set_f, evaluation_f, distance=1, steps=100, normalization=None) -> np.ndarray:
    """
    Returns an approximation of the reward attained by the agent in its environment,
    along a planar subspace of the internal model's parameter space defined by a
    start point and two randomly sampled directions.

    That is, given an agent, a neural network model whose parameters define a point
    in parameter space, and a distance, the expected reward for the agent is computed
    at the given number of steps along the plane defined by the two random directions,
    from the start point up to the maximum distance in both directions.

    Note that the dimensionality of the model parameters has an impact on the expected
    length of a uniformly sampled vector in parameter space. That is, the more parameters
    a model has, the longer the distance in the random vector's direction should be,
    in order to see meaningful change in individual parameters. Normalizing the
    direction vector according to the model's current parameter values, which is supported
    through the 'normalization' parameter, helps reduce the impact of the distance
    parameter. In future releases, the distance parameter will refer to the maximum change
    in an individual parameter, rather than the length of the random direction vector.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss. The model setting function has to
    be of the form

        model_set_function(agent, model)

    and must specify a procedure by which the agent's internal parametric model is set
    to be the one given as an argument.

    :param agent: agent to evaluate
    :param model: parametric model used in agent
    :param model_set_f: a function that defines how to set the agent's internal model
    :param evaluation_f: a function that defines how to compute the expected reward of an agent
    :param distance: maximum distance along the line in parameter space
    :param steps: number of steps to divide the distance into
    :param normalization: normalization method for the direction vector
    :return: 1-d array of return values along the line connecting start and end models
    """
    return loss_landscapes.compute.compute_random_plane(model, evaluation_f, distance, steps, normalization,
                                                        agent, model_set_f)


def planar_interpolation(agent, model_start, model_end_one, model_end_two, model_set_f, evaluation_f,
                         steps=100) -> np.ndarray:
    """
    Returns an approximation of the reward attained by the agent in its environment,
    along a planar subspace of the internal model's parameter space defined by a
    start point and two end points (the two directions defining the plane are taken
    to be the displacement vectors from the start to the two end points).

    That is, given an agent, and two neural network models whose parameters define points
    in parameter space, the expected reward for the agent is computed at the given number
    of steps along the plane defined by the three points.

    The evaluation function supplied has to be of the form

        evaluation_function(model)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss. The model setting function has to
    be of the form

        model_set_function(agent, model)

    and must specify a procedure by which the agent's internal parametric model is set
    to be the one given as an argument.

    :param agent: agent to evaluate
    :param model_start: parametric model used in agent, represents origin of plane
    :param model_end_one: parametric model used in agent, represents end point of first direction vector
    :param model_end_two: parametric model used in agent, represents end point of second direction vector
    :param model_set_f: a function that defines how to set the agent's internal model
    :param evaluation_f: a function that defines how to compute the expected reward of an agent
    :param steps: number of steps to divide the distance into
    :return: 1-d array of return values along the line connecting start and end models
    """
    return loss_landscapes.compute.compute_planar_interpolation(model_start, model_end_one, model_end_two, evaluation_f,
                                                                steps, agent, model_set_f)
