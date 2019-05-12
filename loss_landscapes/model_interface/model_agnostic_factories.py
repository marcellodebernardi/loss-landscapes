import loss_landscapes.model_interface.model_wrapper as model_wrapper
import loss_landscapes.model_interface.model_tensor as model_tensor
import loss_landscapes.model_interface.torch.torch_wrappers as torch_wrappers
import loss_landscapes.model_interface.torch.torch_tensor as torch_tensor


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


def wrap_model(model) -> model_wrapper.ModelWrapper:
    """
    Returns an appropriate wrapper for the given model. For example, if the
    model is a PyTorch model, returns a TorchModelWrapper for the model.
    :param model: model to wrap
    :return: appropriate wrapper for model
    """
    # difficult in Python to type check without importing the relevant libraries - but we don't
    # want to import every supported DL library just to type check (for example, TF takes a few seconds
    # to import). So instead we use a 'hack': we check if the model object has a library-specific
    # method. For example, if the model has a method state_dict(), we assume it's a PyTorch method.
    # Naturally this method is not robust, as a user could very well add a method 'state_dict' to
    # a Keras model. For now, we assume this would happen infrequently. todo this needs to be fixed

    # try torch model
    try:
        return torch_wrappers.TorchNamedParameterWrapper(model)
    except AttributeError:
        raise TypeError('Model type not supported')
