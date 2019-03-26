import torch
import torch.nn
import numpy as np
import backends.torch.ops as ops


def line(model, direction, distance, steps, evaluation) -> np.ndarray:
    """
    Computes loss for discretized intervals on an one-dimensional parameter line.

    Args:
        model: converter that wraps the model and updates its parameters
        direction: direction vector in parameter space
        distance: distance along direction to reach
        steps: how many steps along direction to reach distance
        evaluation: model evaluation function
    """
    # set all parameters in the model to not require gradients, which allows
    # using torch operations on the parameters
    ops.unrequire_grad__(model)

    losses = []
    direction_step = ops.scalar_multiplication(direction, distance / steps)

    for _ in range(steps):
        ops.vector_addition__(model.parameters(), direction_step)
        losses.append(evaluation(model))

    return np.array(losses)


def plane(model, direction_one, direction_two, distance_one, distance_two, steps, evaluation, center) -> np.ndarray:
    """
    Computes loss for discretized intervals on a two-dimensional parameter plane.

    Args:
        model: model for which parameters are to be updated
        direction_one: first direction vector in parameter space
        direction_two: second direction vector in parameter space
        distance_one: distance along direction to reach
        distance_two: distance along direction to reach
        steps: how many steps along direction to reach distance
        evaluation: model evaluation function
        center: whether the start point should be the top-left or central point of the plane
    """
    # set all parameters in the model to not require gradients, which allows
    # using torch operations on the parameters
    ops.unrequire_grad__(model)

    losses = []
    direction_one_step = ops.scalar_multiplication(direction_one, distance_one / steps)
    direction_two_step = ops.scalar_multiplication(direction_two, distance_two / steps)
    total_displacement_two = ops.scalar_multiplication(direction_two, distance_two)

    for _ in range(steps):
        inner_losses = []
        for _ in range(steps):
            ops.vector_addition__(model.parameters(), direction_two_step)
            inner_losses.append(evaluation(model))

        losses.append(inner_losses)
        ops.vector_subtraction__(model.parameters(), total_displacement_two)
        ops.vector_addition__(model.parameters(), direction_one_step)

    return np.array(losses)


def sample_uniform_like(source_parameters, unit_vector=True) -> list:
    new_vector = []

    for p in source_parameters:
        new_vector.append(torch.nn.Parameter(torch.rand_like(p, requires_grad=False), requires_grad=False))

    return ops.unit_vector__(new_vector) if unit_vector else new_vector


def get_normalized_vector(direction, model_parameters, norm_type) -> list:
    if norm_type == 'filter':
        return _get_filter_normalized(direction, model_parameters)
    elif norm_type == 'layer':
        return _get_layer_normalized(direction, model_parameters)
    elif norm_type == 'model':
        return _get_model_normalized(direction, model_parameters)


def _get_filter_normalized(direction, model_parameters) -> list:
    """ Applies filter normalization to a direction vector """
    normalized_vector = []

    # iterate over every layer's every filter
    for layer_idx in range(len(model_parameters)):
        normalized_direction_layer = direction[layer_idx].clone().detach()
        model_layer = model_parameters[layer_idx]

        # todo not sure if this works for all types of layers, conv layers may not be 2-dimensional
        for dir_filter, mod_filter in zip(normalized_direction_layer, model_layer):
            # for each filter, the corresponding new filter is obtained by the equation in the paper
            frob_norm_dir = np.linalg.norm(dir_filter.numpy(), ord='fro')
            frob_norm_mod = np.linalg.norm(mod_filter.numpy(), ord='fro')

            ops.scalar_multiplication__(ops.scalar_division__(dir_filter, frob_norm_dir), frob_norm_mod)

        normalized_vector.append(normalized_direction_layer)

    return normalized_vector


def _get_layer_normalized(direction, model_parameters) -> list:
    pass


def _get_model_normalized(direction, model_parameters) -> list:
    pass
