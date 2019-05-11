import abc
from loss_landscapes.model_interface.model_tensor import ParameterTensor


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

