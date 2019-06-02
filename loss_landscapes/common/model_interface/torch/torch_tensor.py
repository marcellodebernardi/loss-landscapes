"""
Basic linear algebra operations as defined on lists of numpy arrays.

We can think of these list as a single vectors consisting of all the individual
parameter values. The functions in this module implement basic linear algebra
operations on such lists.

The operations defined in the module follow the PyTorch convention of appending
the '__' suffix to the name of in-place operations.
"""

import math
import numpy as np
import torch
import torch.nn
import common.model_interface.model_tensor as model_tensor
import common.model_interface.torch.torch_vector as torch_vector


class TorchParameterTensor(model_tensor.ParameterTensor):
    def __init__(self, parameters: list):
        if not isinstance(parameters, list) and all(isinstance(p, torch.nn.parameter.Parameter) for p in parameters):
            raise AttributeError('Argument to ParameterVector is not a list of numpy arrays.')

        self.parameters = parameters

    def __len__(self) -> int:
        return len(self.parameters)

    def numel(self) -> int:
        return sum(p.numel() for p in self.parameters)

    def __getitem__(self, index) -> torch.nn.Parameter:
        return self.parameters[index]

    def __eq__(self, other) -> bool:
        if not isinstance(other, TorchParameterTensor) or len(self) != len(other):
            return False
        else:
            return all(torch.equal(p_self, p_other) for p_self, p_other in zip(self.parameters, other.parameters))

    def __add__(self, other) -> 'TorchParameterTensor':
        return TorchParameterTensor([self[idx] + other[idx] for idx in range(len(self))])

    def __radd__(self, other) -> 'TorchParameterTensor':
        return self.__add__(other)

    def add_(self, vector):
        for idx in range(len(self)):
            self.parameters[idx] += vector[idx]

    def __sub__(self, other) -> 'TorchParameterTensor':
        return TorchParameterTensor([self[idx] - other[idx] for idx in range(len(self))])

    def __rsub__(self, other) -> 'TorchParameterTensor':
        return self.__sub__(other)

    def sub_(self, vector):
        for idx in range(len(self)):
            self.parameters[idx] -= vector[idx]

    def __mul__(self, scalar) -> 'TorchParameterTensor':
        return TorchParameterTensor([self[idx] * scalar for idx in range(len(self))])

    def __rmul__(self, scalar) -> 'TorchParameterTensor':
        return self.__mul__(scalar)

    def mul_(self, scalar):
        for idx in range(len(self)):
            self.parameters[idx] *= scalar

    def __truediv__(self, scalar) -> 'TorchParameterTensor':
        return TorchParameterTensor([self[idx] / scalar for idx in range(len(self))])

    def truediv_(self, scalar):
        for idx in range(len(self)):
            self.parameters[idx] /= scalar

    def __floordiv__(self, scalar) -> 'TorchParameterTensor':
        return TorchParameterTensor([self[idx] // scalar for idx in range(len(self))])

    def floordiv_(self, scalar):
        for idx in range(len(self)):
            self.parameters[idx] //= scalar

    def __matmul__(self, other) -> 'TorchParameterTensor':
        raise NotImplementedError()

    def model_normalize_(self, ref_point=None, order=2):
        norm = ref_point.model_norm(order) if ref_point is not None \
            else self.model_norm(order)
        for parameter in self.parameters:
            parameter /= norm

    def layer_normalize_(self, ref_point=None, order=2):
        # in-place normalize each parameter
        for layer_idx, parameter in enumerate(self.parameters, 0):
            norm = ref_point.layer_norm(layer_idx, order) if ref_point is not None \
                else self.layer_norm(layer_idx, order)
            parameter /= norm

    def filter_normalize_(self, ref_point=None, order=2):
        for l in range(len(self.parameters)):
            for f in range(len(self.parameters[l])):
                norm = ref_point.filter_norm((l, f), order) if ref_point is not None \
                    else self.filter_norm((l, f), order)
                self.parameters[l][f] /= norm

    def model_norm(self, order=2) -> float:
        # L-n norm of model where we treat the model as a flat vector
        return math.pow(sum([
            torch.pow(layer, order).sum().item()
            for layer in self.parameters
        ]), 1.0 / order)

    def layer_norm(self, index, order=2) -> float:
        # L-n norms of layer where we treat each layer as a flat vector
        return math.pow(torch.pow(self.parameters[index], order).sum().item(), 1.0 / order)

    def filter_norm(self, index, order=2) -> float:
        # L-n norm of each filter where we treat each layer as a flat vector
        return math.pow(torch.pow(self.parameters[index[0]][index[1]], order).sum().item(), 1.0 / order)

    def as_numpy(self) -> np.ndarray:
        return np.concatenate([p.numpy().flatten() for p in self.parameters])

    def as_vector(self) -> torch_vector.TorchParameterVector:
        raise NotImplementedError()  # todo

    def _get_parameters(self) -> list:
        return self.parameters


def rand_u_like(example_vector) -> TorchParameterTensor:
    new_vector = []

    for param in example_vector:
        new_vector.append(torch.rand(size=param.size(), dtype=example_vector[0].dtype))

    return TorchParameterTensor(new_vector)
