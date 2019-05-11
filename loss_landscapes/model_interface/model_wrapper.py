import abc
from loss_landscapes.model_interface.model_tensor import ParameterTensor
from loss_landscapes.model_interface.torch.torch_wrappers import TorchModelWrapper


class ModelWrapper(abc.ABC):
    @abc.abstractmethod
    def get_model(self):
        """
        Return the model encapsulated in this wrapper.
        :return: wrapped model
        """
        pass

    @abc.abstractmethod
    def get_parameters(self) -> ParameterTensor:
        """
        Return a deep copy of the parameters made accessible by this wrapper.
        :return: deep copy of accessible model parameters
        """
        pass

    @abc.abstractmethod
    def set_parameters(self, new_parameters: ParameterTensor):
        """
        Sets the parameters of the wrapped model to the given ParameterVector.
        :param new_parameters: ParameterVector of new parameters
        :return: none
        """
        pass


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

