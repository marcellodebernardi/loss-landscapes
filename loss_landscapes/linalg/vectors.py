import math
import numpy as np
from state_representation import LayeredVector


def addition(vector_a: LayeredVector, vector_b: LayeredVector) -> LayeredVector:
    if len(vector_a) != len(vector_b):
        raise ValueError('Parameter vector lists do not have the same length.')

    new_vector = LayeredVector()

    for layer_idx in range(len(vector_a)):
        new_vector.add_layer(vector_a[layer_idx] + vector_b[layer_idx])

    return new_vector


def subtraction(vector_a: LayeredVector, vector_b: LayeredVector) -> LayeredVector:
    if len(vector_a) != len(vector_b):
        raise ValueError('Parameter vector lists do not have the same length.')

    new_vector = LayeredVector()

    for layer_idx in range(len(vector_a)):
        new_vector.add_layer(vector_a[layer_idx] - vector_b[layer_idx])

    return new_vector


def scalar_multiplication(vector: LayeredVector, scalar) -> LayeredVector:
    new_vector = LayeredVector()

    for layer_idx in range(len(vector)):
        new_vector.add_layer(vector[layer_idx] * scalar)

    return new_vector


def scalar_division(vector: LayeredVector, scalar) -> LayeredVector:
    return scalar_multiplication(vector, 1/scalar)


def l2_norm(parameters: LayeredVector) -> float:
    sum_of_squares = 0

    for layer in parameters:
        sum_of_squares += np.power(layer, 2).sum()

    return math.sqrt(sum_of_squares)


def unit_vector(parameters: LayeredVector) -> LayeredVector:
    return scalar_division(parameters, l2_norm(parameters))
