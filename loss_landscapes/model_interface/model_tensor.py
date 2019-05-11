import copy
from abc import ABC, abstractmethod
from loss_landscapes.model_interface.model_vector import ParameterVector
from loss_landscapes.model_interface.torch.torch_tensor import TorchNamedParameterTensor
from loss_landscapes.model_interface.torch.torch_tensor import rand_u_like as torch_rand_u_like


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
    def __getitem__(self, index):
        """
        Returns the tensor of the layer at the given index.
        :param index: layer index
        :return: tensor of layer
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
    def model_normalize_(self, order=2):
        """
        In-place model-wise normalization of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        pass

    @abstractmethod
    def layer_normalize_(self, order=2):
        """
        In-place layer-wise normalization of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        pass

    @abstractmethod
    def filter_normalize_(self, order=2):
        """
        In-place filter-wise normalization of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: none
        """
        pass

    @abstractmethod
    def _model_norm(self, order=2) -> float:
        """
        Returns the model-wise L-norm of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: L-norm of tensor
        """
        pass

    @abstractmethod
    def _layer_norms(self, order=2) -> list:
        """
        Returns a list of layer-wise L-norms of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: list of L-norms of layers
        """
        pass

    @abstractmethod
    def _filter_norms(self, order=2) -> list:
        """
        Returns a 2D list of filter-wise L-norms of the tensor.
        :param order: norm order, e.g. 2 for L2 norm
        :return: list of L-norms of filters
        """
        pass

    @abstractmethod
    def as_numpy_list(self) -> list:
        """
        Returns the tensor as a list of numpy arrays.
        :return: list of numpy arrays
        """
        pass

    @abstractmethod
    def as_vector(self) -> ParameterVector:
        """
        Returns a flattened view of the tensor which shares the underlying elements. Note the
        aliasing of the tensor/vector elements.
        :return: flat view of the tensor
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


def rand_u_like(example_tensor: ParameterTensor) -> ParameterTensor:
    """
    Constructively return a ParameterTensor with the same structure as the provided
    example tensor, where each element of the tensor is randomly sampled from a
    uniform [0, 1] distribution.
    :param example_tensor: shape of this tensor is copied
    :return: randomly sampled tensor with same shape as example tensor
    """
    if isinstance(example_tensor, TorchNamedParameterTensor):
        return torch_rand_u_like(example_tensor)
    else:
        raise TypeError('Input of type ' + str(type(example_tensor)) + ' is not recognized.')


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
