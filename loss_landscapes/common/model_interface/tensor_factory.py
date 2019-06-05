import loss_landscapes.common.model_interface.model_tensor as model_tensor
import loss_landscapes.common.model_interface.torch.torch_tensor as torch_tensor


def rand_u_like(example_tensor: model_tensor.ParameterTensor) -> model_tensor.ParameterTensor:
    """
    Constructively return a ParameterTensor with the same structure as the provided
    example tensor, where each element of the tensor is randomly sampled from a
    uniform [0, 1] distribution.
    :param example_tensor: shape of this tensor is copied
    :return: randomly sampled tensor with same shape as example tensor
    """
    if isinstance(example_tensor, torch_tensor.TorchParameterTensor):
        return torch_tensor.rand_u_like(example_tensor)
    else:
        raise TypeError('Input of type ' + str(type(example_tensor)) + ' is not recognized.')
