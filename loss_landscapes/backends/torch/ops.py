""" Defines basic linear algebra operations on lists of torch.nn.Parameter objects. """


import math
import torch
import torch.nn


def l2_norm(vector) -> float:
    length = 0

    for p in vector:
        length += torch.pow(p.clone().detach(), 2).sum().item()

    return math.sqrt(length)


def vector_addition(vector_a, vector_b) -> list:
    new_vector = []

    for idx in range(len(vector_a)):
        new_vector.append(vector_a[idx].clone().detach() + vector_b[idx].clone().detach())

    return new_vector


def vector_subtraction(vector_a, vector_b) -> list:
    new_vector = []

    for idx in range(len(vector_a)):
        new_vector.append(vector_a[idx].clone().detach() - vector_b[idx].clone().detach())

    return new_vector


def scalar_multiplication(vector_a, scalar) -> list:
    new_vector = []

    for idx in range(len(vector_a)):
        new_vector.append(vector_a[idx].clone().detach() * scalar)

    return new_vector


def scalar_division(vector, scalar) -> list:
    return scalar_multiplication(vector, 1/scalar)


def unit_vector(vector) -> list:
    return scalar_division(vector, l2_norm(vector))


def vector_addition__(target_vector, vector):
    for p_target, p in zip(target_vector, vector):
        p_target += p


def vector_subtraction__(target_vector, vector):
    for p_target, p in zip(target_vector, vector):
        p_target += p


def scalar_multiplication__(target_vector, scalar):
    for p in target_vector:
        p *= scalar


def scalar_division__(target_vector, scalar):
    for p in target_vector:
        p /= scalar


def unit_vector__(target_vector):
    length = l2_norm(target_vector)
    for p in target_vector:
        p /= length


def unrequire_grad__(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False
