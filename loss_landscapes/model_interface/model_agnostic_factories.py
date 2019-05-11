from loss_landscapes.model_interface.model_wrapper import ModelWrapper
from loss_landscapes.model_interface.model_tensor import ParameterTensor
from loss_landscapes.model_interface.torch.torch_wrappers import TorchModelWrapper
from loss_landscapes.model_interface.torch.torch_tensor import TorchNamedParameterTensor
from loss_landscapes.model_interface.torch.torch_tensor import rand_u_like as torch_rand_u_like


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


def wrap_model(model) -> ModelWrapper:
    """
    Returns an appropriate wrapper for the given model. For example, if the
    model is a PyTorch model, returns a TorchModelWrapper for the model.
    :param model: model to wrap
    :return: appropriate wrapper for model
    """
    if 'torch' in str(type(model)):
        return TorchModelWrapper(model)
    else:
        raise TypeError('Model type not supported.')