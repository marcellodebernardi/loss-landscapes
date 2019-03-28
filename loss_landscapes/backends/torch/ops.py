"""
Basic linear algebra operations as defined on lists of torch.nn.Parameter objects.

For PyTorch, we extract the model's parameters by constructing a list out of the
generator returned by torch.nn.Module.parameters(). We can think of this list as
a single vector consisting of all the individual parameter values. The functions
in this module implement basic linear algebra operations on such lists.
"""


import math
import torch
import torch.nn


EMPTY_PARAMETER_LIST = 'The parameter list is empty.'
MISMATCHED_PARAMETER_LENGTH = 'The two parameters lists have mismatched lengths.'


def l2_norm(vector) -> float:
    """ Return the L2 norm of a vector. """
    # check for bad inputs
    _check_inputs(vector)

    length = 0

    for p in vector:
        length += torch.pow(p.clone().detach(), 2).sum().item()

    return math.sqrt(length)


def vector_addition(vector_a, vector_b) -> list:
    """ Construct a new list of Parameters out of the addition of vector_a and vector_b. """
    # check for bad input
    _check_inputs(vector_a, vector_b)

    new_vector = []

    for idx in range(len(vector_a)):
        new_vector.append(vector_a[idx].clone().detach() + vector_b[idx].clone().detach())

    return new_vector


def vector_subtraction(vector_a, vector_b) -> list:
    """ Construct a new list of Parameters out of the subtraction of vector_b from vector_a. """
    # check for bad input
    _check_inputs(vector_a, vector_b)

    new_vector = []

    for idx in range(len(vector_a)):
        new_vector.append(vector_a[idx].clone().detach() - vector_b[idx].clone().detach())

    return new_vector


def scalar_multiplication(vector, scalar) -> list:
    """ Construct a new list of Parameters out of the multiplication of a given vector by a scalar. """
    # check for bad input
    _check_inputs(vector)

    new_vector = []

    for idx in range(len(vector)):
        new_vector.append(vector[idx].clone().detach() * scalar)

    return new_vector


def scalar_division(vector, scalar) -> list:
    """ Construct a new list of Parameters out of the division of a given vector by a scalar. """
    # check for bad input
    _check_inputs(vector)

    return scalar_multiplication(vector, 1/scalar)


def unit_vector(vector) -> list:
    """ Construct a unit vector pointing in the direction of the given vector. """
    # check for bad input
    _check_inputs(vector)

    return scalar_division(vector, l2_norm(vector))


def vector_addition__(target_vector, vector):
    """ Modify the target vector in-place by adding the other vector to it. """
    # check for bad input
    _check_inputs(target_vector, vector)

    for p_target, p in zip(target_vector, vector):
        p_target.add_(p)


def vector_subtraction__(target_vector, vector):
    """ Modify the target vector in-place by subtracting the other vector from it. """
    # check for bad input
    _check_inputs(target_vector, vector)

    for p_target, p in zip(target_vector, vector):
        p_target -= p


def scalar_multiplication__(target_vector, scalar):
    """ Modify the target vector in-place by multiplying it by the given scalar. """
    # check for bad input
    _check_inputs(target_vector)

    for p in target_vector:
        p.mul_(scalar)


def scalar_division__(target_vector, scalar):
    """ Modify the target vector in-place by dividing it by the given scalar. """
    # check for bad input
    _check_inputs(target_vector)

    for p in target_vector:
        p.div_(scalar)


def unit_vector__(target_vector):
    """ Modify the target vector in-place, turning it into a unit vector pointing in the same direction. """
    # check for bad input
    _check_inputs(target_vector)
    scalar_division__(target_vector, l2_norm(target_vector))


def unrequire_grad__(model: torch.nn.Module):
    """ Modify the given torch model by specifying that its parameters do not require autograd. """
    for p in model.parameters():
        p.requires_grad = False


def _check_inputs(*args):
    """ Raises an error if the given vectors are not compatible. """
    # input may be a generator in the case of model.parameters() being used in-place
    vectors = [list(x) for x in args]

    length = len(vectors[0])

    if length == 0:
        raise ValueError(EMPTY_PARAMETER_LIST)

    for vector in vectors[1:]:
        if len(vector) != length:
            raise ValueError(MISMATCHED_PARAMETER_LENGTH)
