import abc
import loss_landscapes.model_interface.model_tensor as model_tensor


class ModelWrapper(abc.ABC):
    @abc.abstractmethod
    def get_model(self):
        """
        Return the model encapsulated in this wrapper.
        :return: wrapped model
        """
        pass

    @abc.abstractmethod
    def get_parameters(self) -> model_tensor.ParameterTensor:
        """
        Return a deep copy of the parameters made accessible by this wrapper.
        :return: deep copy of accessible model parameters
        """
        pass

    @abc.abstractmethod
    def set_parameters(self, new_parameters: model_tensor.ParameterTensor):
        """
        Sets the parameters of the wrapped model to the given ParameterVector.
        :param new_parameters: ParameterVector of new parameters
        :return: none
        """
        pass

