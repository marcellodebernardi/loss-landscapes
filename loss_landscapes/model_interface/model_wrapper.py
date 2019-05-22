import abc
import loss_landscapes.model_interface.model_tensor as model_tensor


class ModelWrapper(abc.ABC):
    @abc.abstractmethod
    def get_model(self):
        """
        Return a reference to the model encapsulated in this wrapper.
        :return: wrapped model
        """
        pass

    @abc.abstractmethod
    def get_parameters(self, deepcopy=False) -> model_tensor.ParameterTensor:
        """
        Return a ParameterTensor wrapping the parameters of the underlying model. The
        parameters can either be returned as a view of the model parameters or as a copy.
        :param deepcopy: whether to view or deepcopy the model parameters
        :return: view or deepcopy of accessible model parameters
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

