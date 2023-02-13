"""
Basic linear algebra operations as defined on the parameter sets of entire models.

We can think of these list as a single vectors consisting of all the individual
parameter values. The functions in this module implement basic linear algebra
operations on such lists.

The operations defined in the module follow the PyTorch convention of appending
the '__' suffix to the name of in-place operations.
"""

import copy
import math
import numpy as np
import torch
import torch.nn


class ModelParameters:
    """
    A ModelParameters object is an abstract view of a model's optimizable parameters as a tensor. This class
    enables the parameters of models of the same 'shape' (architecture) to be operated on as if they were 'real'
    tensors. A ModelParameters object cannot be converted to a true tensor as it is potentially irregularly
    shaped.
    """

    def __init__(self, parameters: list):
        if not isinstance(parameters, list) and all(isinstance(p, torch.Tensor) for p in parameters):
            raise AttributeError('Argument to ModelParameter is not a list of torch.Tensor objects.')

        self.parameters = parameters

    def __len__(self) -> int:
        """
        Returns the number of model layers within the parameter tensor.
        :return: number of layer tensors
        """
        return len(self.parameters)

    def numel(self) -> int:
        """
        Returns the number of elements (i.e. individual parameters) within the tensor.
        Note that this refers to individual parameters, not layers.
        :return: number of elements in tensor
        """
        return sum(p.numel() for p in self.parameters)

    def __getitem__(self, index) -> torch.nn.Parameter:
        """
        Returns the tensor of the layer at the given index.
        :param index: layer index
        :return: tensor of layer
        """
        return self.parameters[index]

    def __eq__(self, other: 'ModelParameters') -> bool:
        """
        Compares this parameter tensor for equality with the argument tensor, using the == operator.
        :param other: the object to compare to
        :return: true if equal
        """
        if not isinstance(other, ModelParameters) or len(self) != len(other):
            return False
        else:
            return all(torch.equal(p_self, p_other) for p_self, p_other in zip(self.parameters, other.parameters))

    def __add__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of addition between this tensor and another.
        :param other: other to add
        :return: self + other
        """
        return ModelParameters([self[idx] + other[idx] for idx in range(len(self))])

    def __radd__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of addition between this tensor and another.
        :param other: model parameters to add
        :return: other + self
        """
        return self.__add__(other)

    def add_(self, other: 'ModelParameters'):
        """
        In-place addition between this tensor and another.
        :param other: model parameters to add
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] += other[idx]

    def __sub__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of subtracting another tensor from this one.
        :param other: model parameters to subtract
        :return: self - other
        """
        return ModelParameters([self[idx] - other[idx] for idx in range(len(self))])

    def __rsub__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of subtracting this tensor from another one.
        :param other: other to subtract from
        :return: other - self
        """
        return self.__sub__(other)

    def sub_(self, vector: 'ModelParameters'):
        """
        In-place subtraction of another tensor from this one.
        :param vector: other to subtract
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] -= vector[idx]

    def __mul__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of multiplying this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: self * scalar
        """
        return ModelParameters([self[idx] * scalar for idx in range(len(self))])

    def __rmul__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of multiplying this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: scalar * self
        """
        return self.__mul__(scalar)

    def mul_(self, scalar):
        """
        In-place multiplication of this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] *= scalar

    def __truediv__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of true-dividing this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: scalar / self
        """
        return ModelParameters([self[idx] / scalar for idx in range(len(self))])

    def truediv_(self, scalar):
        """
        In-place true-division of this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] /= scalar

    def __floordiv__(self, scalar) -> 'ModelParameters':
        """
        Constructively returns the result of floor-dividing this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: scalar // self
        """
        return ModelParameters([self[idx] // scalar for idx in range(len(self))])

    def floordiv_(self, scalar):
        """
        In-place floor-division of this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: none
        """
        for idx in range(len(self)):
            self.parameters[idx] //= scalar

    def __matmul__(self, other: 'ModelParameters') -> 'ModelParameters':
        """
        Constructively returns the result of tensor-multiplication of this tensor by another tensor.
        :param other: other tensor
        :return: self @ tensor
        """
        raise NotImplementedError()

    def dot(self, other: 'ModelParameters') -> float:
        """
        Returns the vector dot product of this ModelParameters vector and the given other vector.
        :param other: other ModelParameters vector
        :return: dot product of self and other
        """
        param_products = []
        for idx in range(len(self.parameters)):
            param_products.append((self.parameters[idx] * other.parameters[idx]).sum().item())
        return sum(param_products)

    def model_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place model-wise normalization of the tensor.
        :param ref_point: use this model's norm, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        for parameter in self.parameters:
            parameter *= (ref_point.model_norm(order) / self.model_norm())

    def layer_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place layer-wise normalization of the tensor.
        :param ref_point: use this model's layer norms, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        # in-place normalize each parameter
        for layer_idx, parameter in enumerate(self.parameters, 0):
            parameter *= (ref_point.layer_norm(layer_idx, order) / self.layer_norm(layer_idx, order))

    def filter_normalize_(self, ref_point: 'ModelParameters', order=2):
        """
        In-place filter-wise normalization of the tensor.
        :param ref_point: use this model's filter norms, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        for l in range(len(self.parameters)):
            # normalize one-dimensional bias vectors
            if len(self.parameters[l].size()) == 1:
                self.parameters[l] *= (ref_point.parameters[l].norm(order) / self.parameters[l].norm(order))
            # normalize two-dimensional weight vectors
            for f in range(len(self.parameters[l])):
                self.parameters[l][f] *= ref_point.filter_norm((l, f), order) / (self.filter_norm((l, f), order))

    def model_norm(self, order=2) -> float:
        """
        Returns the model-wise L-norm of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: L-norm of tensor
        """
        # L-n norm of model where we treat the model as a flat other
        return math.pow(sum([
            torch.pow(layer, order).sum().item()
            for layer in self.parameters
        ]), 1.0 / order)

    def layer_norm(self, index, order=2) -> float:
        """
        Returns a list of layer-wise L-norms of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :param index: layer index
        :return: list of L-norms of layers
        """
        # L-n norms of layer where we treat each layer as a flat other
        return math.pow(torch.pow(self.parameters[index], order).sum().item(), 1.0 / order)

    def filter_norm(self, index, order=2) -> float:
        """
        Returns a 2D list of filter-wise L-norms of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :param index: tuple with layer index and filter index
        :return: list of L-norms of filters
        """
        # L-n norm of each filter where we treat each layer as a flat other
        return math.pow(torch.pow(self.parameters[index[0]][index[1]], order).sum().item(), 1.0 / order)

    def as_numpy(self) -> np.ndarray:
        """
        Returns the tensor as a flat numpy array.
        :return: a numpy array
        """
        return np.concatenate([p.numpy().flatten() for p in self.parameters])

    def _get_parameters(self) -> list:
        """
        Returns a reference to the internal parameter data in whatever format used by the source model.
        :return: reference to internal parameter data
        """
        return self.parameters


def rand_u_like(example_vector: ModelParameters) -> ModelParameters:
    """
    Create a new ModelParameters object of size and shape compatible with the given
    example vector, such that the values in the ModelParameter are uniformly distributed
    in the range [0,1].
    :param example_vector: defines by example the size and shape the new vector will have
    :return: new vector with uniformly distributed values
    """
    new_vector = []

    for param in example_vector:
        new_vector.append(torch.rand(size=param.size(), dtype=example_vector[0].dtype).to(param.device))

    return ModelParameters(new_vector)


def rand_n_like(example_vector: ModelParameters) -> ModelParameters:
    """
    Create a new ModelParameters object of size and shape compatible with the given
    example vector, such that the values in the ModelParameter are normally distributed
    as N(0,1).
    :param example_vector: defines by example the size and shape the new vector will have
    :return: new vector with normally distributed values
    """
    new_vector = []

    for param in example_vector:
        new_vector.append(torch.randn(size=param.size(), dtype=example_vector[0].dtype).to(param.device))

    return ModelParameters(new_vector)


def orthogonal_to(vector: ModelParameters) -> ModelParameters:
    """
    Create a new ModelParameters object of size and shape compatible with the given
    example vector, such that the two vectors are very nearly orthogonal.
    :param vector: original vector
    :return: new vector that is very nearly orthogonal to original vector
    """
    new_vector = rand_u_like(vector)
    new_vector = new_vector - new_vector.dot(vector) * vector / math.pow(vector.model_norm(2), 2)
    return new_vector


def add(vector_a: ModelParameters, vector_b: ModelParameters) -> ModelParameters:
    return vector_a + vector_b


def sub(vector_a: ModelParameters, vector_b: ModelParameters) -> ModelParameters:
    return vector_a - vector_b


def mul(vector: ModelParameters, scalar) -> ModelParameters:
    return vector * scalar


def truediv(vector: ModelParameters, scalar) -> ModelParameters:
    return vector / scalar


def floordiv(vector: ModelParameters, scalar) -> ModelParameters:
    return vector // scalar


def filter_normalize(tensor, order=2) -> ModelParameters:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.filter_normalize_(order)
    return new_tensor


def layer_normalize(tensor, order) -> ModelParameters:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.layer_normalize_(order)
    return new_tensor


def model_normalize(tensor, order) -> ModelParameters:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.model_normalize_(order)
    return new_tensor
