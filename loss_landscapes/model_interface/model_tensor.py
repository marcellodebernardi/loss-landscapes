import copy
from abc import ABC, abstractmethod
import numpy as np


class ParameterTensor(ABC):
    """
    A ParameterTensor represents a layer-wise (and therefore potentially irregularly shaped)
    tensor containing all of a deep learning model's parameters. The class defines linear
    algebra operations.
    """
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of model layers within the parameter tensor.
        :return: number of layer tensors
        """
        pass

    @abstractmethod
    def numel(self) -> int:
        """
        Returns the number of elements (i.e. individual parameters) within the tensor.
        Note that this refers to individual parameters, not layers.
        :return: number of elements in tensor
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        Returns the tensor of the layer at the given index.
        :param index: layer index
        :return: tensor of layer
        """
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        """
        Compares this parameter tensor for equality with the argument tensor, using the == operator.

        :param other: the object to compare to
        :return: true if equal
        """
        pass

    @abstractmethod
    def __add__(self, vector) -> 'ParameterTensor':
        """
        Constructively returns the result of addition between this tensor and another.
        :param vector: vector to add
        :return: self + other
        """
        pass

    @abstractmethod
    def __radd__(self, vector) -> 'ParameterTensor':
        """
        Constructively returns the result of addition between this tensor and another.
        :param vector: vector to add
        :return: other + self
        """
        pass

    @abstractmethod
    def add_(self, vector):
        """
        In-place addition between this tensor and another.
        :param vector: vector to add
        :return: none
        """
        pass

    @abstractmethod
    def __sub__(self, vector) -> 'ParameterTensor':
        """
        Constructively returns the result of subtracting another tensor from this one.
        :param vector: vector to subtract
        :return: self - other
        """
        pass

    @abstractmethod
    def __rsub__(self, vector) -> 'ParameterTensor':
        """
        Constructively returns the result of subtracting this tensor from another one.
        :param vector: vector to subtract from
        :return: other - self
        """
        pass

    @abstractmethod
    def sub_(self, vector):
        """
        In-place subtraction of another tensor from this one.
        :param vector: vector to subtract
        :return: none
        """
        pass

    @abstractmethod
    def __mul__(self, scalar) -> 'ParameterTensor':
        """
        Constructively returns the result of multiplying this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: self * scalar
        """
        pass

    @abstractmethod
    def __rmul__(self, scalar) -> 'ParameterTensor':
        """
        Constructively returns the result of multiplying this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: scalar * self
        """
        pass

    @abstractmethod
    def mul_(self, scalar):
        """
        In-place multiplication of this tensor by a scalar.
        :param scalar: scalar to multiply by
        :return: none
        """
        pass

    @abstractmethod
    def __truediv__(self, scalar) -> 'ParameterTensor':
        """
        Constructively returns the result of true-dividing this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: scalar / self
        """
        pass

    @abstractmethod
    def truediv_(self, scalar):
        """
        In-place true-division of this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: none
        """
        pass

    @abstractmethod
    def __floordiv__(self, scalar) -> 'ParameterTensor':
        """
        Constructively returns the result of floor-dividing this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: scalar // self
        """
        pass

    @abstractmethod
    def floordiv_(self, scalar):
        """
        In-place floor-division of this tensor by a scalar.
        :param scalar: scalar to divide by
        :return: none
        """
        pass

    @abstractmethod
    def __matmul__(self, other) -> 'ParameterTensor':
        """
        Constructively returns the result of tensor-multiplication of this tensor by another tensor.
        :param other: other tensor
        :return: self @ tensor
        """
        pass

    @abstractmethod
    def model_normalize_(self, ref_point=None, order=2):
        """
        In-place model-wise normalization of the tensor.
        :param ref_point: use this model's norm, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        pass

    @abstractmethod
    def layer_normalize_(self, ref_point=None, order=2):
        """
        In-place layer-wise normalization of the tensor.
        :param ref_point: use this model's layer norms, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        pass

    @abstractmethod
    def filter_normalize_(self, ref_point=None, order=2):
        """
        In-place filter-wise normalization of the tensor.
        :param ref_point: use this model's filter norms, if given
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        pass

    @abstractmethod
    def model_norm(self, order=2) -> float:
        """
        Returns the model-wise L-norm of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: L-norm of tensor
        """
        pass

    @abstractmethod
    def layer_norm(self, index: int, order=2) -> float:
        """
        Returns a list of layer-wise L-norms of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :param index: layer index
        :return: list of L-norms of layers
        """
        pass

    @abstractmethod
    def filter_norm(self, index: tuple, order=2) -> float:
        """
        Returns a 2D list of filter-wise L-norms of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :param index: tuple with layer index and filter index
        :return: list of L-norms of filters
        """
        pass

    @abstractmethod
    def as_numpy(self) -> np.ndarray:
        """
        Returns the tensor as a flat numpy array.
        :return: a numpy array
        """
        pass

    @abstractmethod
    def _get_parameters(self):
        """
        Returns a reference to the internal parameter data in whatever format used by the source model.
        :return: reference to internal parameter data
        """
        pass


def add(vector_a: ParameterTensor, vector_b: ParameterTensor) -> ParameterTensor:
    return vector_a + vector_b


def sub(vector_a: ParameterTensor, vector_b: ParameterTensor) -> ParameterTensor:
    return vector_a - vector_b


def mul(vector: ParameterTensor, scalar) -> ParameterTensor:
    return vector * scalar


def truediv(vector: ParameterTensor, scalar) -> ParameterTensor:
    return vector / scalar


def floordiv(vector: ParameterTensor, scalar) -> ParameterTensor:
    return vector // scalar


def filter_normalize(tensor, order=2) -> ParameterTensor:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.filter_normalize_(order)
    return new_tensor


def layer_normalize(tensor, order) -> ParameterTensor:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.layer_normalize_(order)
    return new_tensor


def model_normalize(tensor, order) -> ParameterTensor:
    new_tensor = copy.deepcopy(tensor)
    new_tensor.model_normalize_(order)
    return new_tensor
