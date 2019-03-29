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

    :param agent:
    :param model:
    :param model_set_f:
    :param evaluation_f:
    :param distance:
    :param steps:
    :param normalization:
    :return:
    """
    return loss_landscapes.compute.compute_random_line(model, evaluation_f, distance, steps, normalization, agent,
                                                       model_set_f)


def linear_interpolation(agent, model_start, model_end, model_set_f, evaluation_f, steps=100) -> np.ndarray:
    """

    :param agent:
    :param model_start:
    :param model_end:
    :param model_set_f:
    :param evaluation_f:
    :param steps:
    :return:
    """
    return loss_landscapes.compute.compute_linear_interpolation(model_start, model_end, evaluation_f, steps, agent,
                                                                model_set_f)


def random_plane(agent, model_start, model_set_f, evaluation_f, distance=1, steps=100,
                 normalization=None) -> np.ndarray:
    """

    :param agent:
    :param model_start:
    :param model_set_f:
    :param evaluation_f:
    :param distance:
    :param steps:
    :param normalization:
    :return:
    """
    return loss_landscapes.compute.compute_random_plane(model_start, evaluation_f, distance, steps, normalization,
                                                        agent, model_set_f)


def planar_interpolation(agent, model_start, model_end_one, model_end_two, model_set_f, evaluation_f,
                         steps=100) -> np.ndarray:
    """

    :param agent:
    :param model_start:
    :param model_end_one:
    :param model_end_two:
    :param model_set_f:
    :param evaluation_f:
    :param steps:
    :return:
    """
    return loss_landscapes.compute.compute_planar_interpolation(model_start, model_end_one, model_end_two, evaluation_f,
                                                                steps, agent, model_set_f)
