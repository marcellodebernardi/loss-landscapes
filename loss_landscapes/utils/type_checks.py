

import torch
from loss_landscapes.model_interface.torch.torch_tensor import TorchParameterTensor


# todo these type checks exist for debugging purposes - remove once not needed
# the library's public interface does not allow client code to create situations
# in which faulty inputs could be passed to the custom tensor operations etc. Such
# bugs can only be introduced by the author of this library in the implementation
# of loss landscape approximations.


def is_scalar(scalar) -> bool:
    return isinstance(scalar, int) or isinstance(scalar, float)


def _are_equisized_torch_tensors(tensor_a, tensor_b) -> bool:
    """ Returns true if the given vectors are fully compatible for addition and subtraction. """
    if not isinstance(tensor_a, TorchParameterTensor):
        raise TypeError('vector_a is of type ' + str(type(tensor_a)) + ', should be ParameterVector.')
    elif not isinstance(tensor_b, TorchParameterTensor):
        raise TypeError('vector_b is of type ' + str(type(tensor_b)) + ', should be ParameterVector.')
    elif not len(tensor_a) == len(tensor_b):
        raise ValueError('Mismatched vector lengths: ' + str(len(tensor_a)) + ' != ' + str(len(tensor_b)))
    elif not all(isinstance(p, torch.nn.parameter.Parameter) for p in tensor_a):
        raise ValueError('Not all elements in vector_a are of type torch.nn.parameter.Parameter.\n'
                         + 'Types found: ' + str([type(element) for element in tensor_a]))
    elif not all(isinstance(p, torch.nn.parameter.Parameter) for p in tensor_b):
        raise ValueError('Not all elements in vector_b are of type torch.nn.parameter.Parameter.\n'
                         + 'Types found: ' + str([type(element) for element in tensor_b]))
    elif not all(pair[0].size() == pair[1].size() for pair in zip(tensor_a._get_parameters(), tensor_b._get_parameters())):
        raise ValueError('Parameters stored in vector_a and vector_b don\'t have matching shapes.\n'
                         + 'vector_a shapes: ' + str([element.size() for element in tensor_a]) + '\n'
                         + 'vector_b shapes: ' + str([element.size() for element in tensor_b]))
    else:
        return True
