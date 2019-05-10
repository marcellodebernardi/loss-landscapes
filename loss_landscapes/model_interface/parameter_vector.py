"""
Basic linear algebra operations as defined on lists of numpy arrays.

We can think of these list as a single vectors consisting of all the individual
parameter values. The functions in this module implement basic linear algebra
operations on such lists.

The operations defined in the module follow the PyTorch convention of appending
the '__' suffix to the name of in-place operations.
"""

import math
import copy
import torch
import torch.nn
import numpy as np

EMPTY_PARAMETER_LIST = 'The parameter list is empty.'
MISMATCHED_PARAMETER_LENGTH = 'The two parameters lists have mismatched lengths.'


def _are_same_size_vectors(vector_a, vector_b) -> bool:
    """ Returns true if the given vectors are fully compatible for addition and subtraction. """
    return isinstance(vector_a, ParameterVector) \
           and isinstance(vector_b, ParameterVector) \
           and len(vector_a) == len(vector_b) \
           and all(isinstance(p, torch.nn.parameter.Parameter) for p in vector_a) \
           and all(isinstance(p, torch.nn.parameter.Parameter) for p in vector_b) \
           and all(pair[0].size() == pair[1].size() for pair in zip([vector_a, vector_b]))


def _is_scalar(scalar) -> bool:
    return isinstance(scalar, int) or isinstance(scalar, float)


class ParameterVector:
    def __init__(self, parameters: list):
        if not isinstance(parameters, list) and all(isinstance(p, torch.nn.parameter.Parameter) for p in parameters):
            raise AttributeError('Argument to ParameterVector is not a list of numpy arrays.')

        self.parameters = parameters

    def get_parameters(self) -> list:
        """ Returns the underlying list instance. """
        return self.parameters

    def __len__(self):
        """ Returns the length of the ParameterVector list. """
        return len(self.parameters)

    def __getitem__(self, index):
        """ Returns the torch tensor at the given index in the Parameter list. """
        return self.parameters[index]

    def __add__(self, other):
        """ Addition of this ParameterVector with another one using the + operator. """
        if not _are_same_size_vectors(self, other):
            raise ValueError('Second input is either not a ParameterVector or does not have the same length as first.')

        result = []
        for idx in range(len(self)):
            result.append(self[idx] + other[idx])
        return ParameterVector(result)

    def __radd__(self, other):
        return self.__add__(other)

    def add_(self, vector):
        """ In-place addition of another ParameterVector to this one. """
        if not _are_same_size_vectors(self, vector):
            raise ValueError('Input is either not a ParameterVector or does not have the same length as first.')

        for idx in range(len(self)):
            self.parameters[idx] += vector[idx]

    def __sub__(self, other):
        """ Subtraction of another ParameterVector from this one using the - operator. """
        if not _are_same_size_vectors(self, other):
            raise ValueError('Second input is either not a ParameterVector or does not have the same length as first.')

        result = []
        for idx in range(len(self)):
            result.append(self[idx] - other[idx])
        return ParameterVector(result)

    def __rsub__(self, other):
        return self.__sub__(other)

    def sub_(self, vector):
        """ In-place subtraction of another ParameterVector from this one. """
        if not _are_same_size_vectors(self, vector):
            raise ValueError('Input is either not a ParameterVector or does not have the same length as first.')

        for idx in range(len(self)):
            self.parameters[idx] -= vector[idx]

    def __mul__(self, other):
        """ Multiplication of this ParameterVector with a scalar using * operator. """
        if not _is_scalar(other):
            raise ValueError('Second input is not a scalar.')

        result = []
        for idx in range(len(self)):
            result.append(self[idx] * other)
        return ParameterVector(result)

    def __rmul__(self, other):
        return self.__mul__(other)

    def mul_(self, scalar):
        if not _is_scalar(scalar):
            raise ValueError('Input is not a scalar.')

        for idx in range(len(self)):
            self.parameters[idx] *= scalar

    def __truediv__(self, other):
        """ Division of this ParameterVector by a scalar, or another ParameterVector, using / operator. """
        if not _is_scalar(other):
            raise ValueError('Second input is not a scalar.')

        result = []
        for idx in range(len(self)):
            result.append(self[idx] / other)
        return ParameterVector(result)

    def truediv_(self, scalar):
        if not _is_scalar(scalar):
            raise ValueError('Input is not a scalar.')

        for idx in range(len(self)):
            self.parameters[idx] /= scalar

    def __floordiv__(self, other):
        """ Floor division of this ParameterVector by a scalar, or another ParameterVector, using // operator. """
        if not _is_scalar(other):
            raise ValueError('Second input is not a scalar.')

        result = []
        for idx in range(len(self)):
            # todo not sure if // is defined for numpy arrays
            result.append(self[idx] // other)
        return ParameterVector(result)

    def floordiv_(self, scalar):
        if not _is_scalar(scalar):
            raise ValueError('Input is not a scalar.')

        for idx in range(len(self)):
            self.parameters[idx] //= scalar

    def __matmul__(self, other):
        """ Matrix multiplication of this ParameterVector with another ParameterVector using @ operator. """
        raise NotImplementedError('Not yet implemented as this use case is not needed as of 09/05/2019.')

    def model_normalize_(self, order=2):
        norm = self._model_norm(order)
        for parameter in self.parameters:
            parameter /= norm

    def layer_normalize_(self, order=2):
        layer_norms = self._layer_norms(order)
        # in-place normalize each parameter
        for layer_idx, parameter in enumerate(self.parameters, 0):
            parameter /= layer_norms[layer_idx]

    def filter_normalize_(self, order=2):
        # filter_norms = self.filter_norm(order)
        raise NotImplementedError()

    def _model_norm(self, order=2):
        n = 0.0
        # collect sum of n-th power of all parameters
        for parameter in self.parameters:
            n += torch.pow(parameter, order).sum().item()
        # return n-th root of the above sum
        return math.pow(n, 1.0 / order)

    def _layer_norms(self, order=2):
        layers = []
        # compute norm of each layer
        for parameter in self.parameters:
            layers.append(np.linalg.norm(parameter, order))
        return layers

    def _filter_norms(self, order=2):
        raise NotImplementedError()

    def numpy(self) -> np.ndarray:
        """
        Returns parameters as a flat (vector-shaped) numpy array.
        :return:  numpy array
        """
        return np.concatenate([p.flatten() for p in self.parameters])


def add(vector_a: ParameterVector, vector_b: ParameterVector):
    if isinstance(vector_a, ParameterVector) and isinstance(vector_b, ParameterVector):
        return vector_a + vector_b
    else:
        raise AttributeError('Both inputs must be of type ParameterVector.')


def sub(vector_a: ParameterVector, vector_b: ParameterVector):
    if isinstance(vector_a, ParameterVector) and isinstance(vector_b, ParameterVector):
        return vector_a - vector_b
    else:
        raise AttributeError('Both inputs must be of type ParameterVector.')


def mul(vector: ParameterVector, scalar):
    if isinstance(vector, ParameterVector) and (isinstance(scalar, int) or isinstance(scalar, float)):
        return vector * scalar
    else:
        raise AttributeError('First argument must be of type ParameterVector, second argument must be int or float.')


def truediv(vector: ParameterVector, scalar):
    if isinstance(vector, ParameterVector) and (isinstance(scalar, int) or isinstance(scalar, float)):
        return vector / scalar
    else:
        raise AttributeError('First argument must be of type ParameterVector, second argument must be int or float.')


def floordiv(vector: ParameterVector, scalar):
    if isinstance(vector, ParameterVector) and (isinstance(scalar, int) or isinstance(scalar, float)):
        return vector // scalar
    else:
        raise AttributeError('First argument must be of type ParameterVector, second argument must be int or float.')


def rand_u_like(example_vector) -> ParameterVector:
    """ Return a vector with uniformly iid sampled dimensions of the same size as example vector. """
    new_vector = []

    for param in example_vector:
        new_vector.append(np.random.uniform(size=np.size(param)))

    return ParameterVector(new_vector)


def filter_normalize(vector, order=2) -> ParameterVector:
    new_vector = copy.deepcopy(vector)
    new_vector.filter_normalize_(order)
    return new_vector


def layer_normalize(vector, order):
    new_vector = copy.deepcopy(vector)
    new_vector.layer_normalize_(order)
    return new_vector


def model_normalize(vector, order):
    new_vector = copy.deepcopy(vector)
    new_vector.model_normalize_(order)
    return new_vector
