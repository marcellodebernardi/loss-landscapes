import abc
import loss_landscapes.common.model_interface.model_tensor as model_tensor


class ModelWrapper(abc.ABC):
    def __init__(self, model, forward_fn):
        self.model = model
        self.forward_fn = forward_fn

    def __call__(self, x):
        """
        Calls the model on the given inputs, and returns the model's output.
        :param x: inputs to the model
        :return: model output
        """
        if self.forward_fn is not None:
            return self.forward_fn(self.model, x)
        else:
            return self.model(x)

    def get_model(self):
        """
        Return a reference to the model encapsulated in this wrapper.
        :return: wrapped model
        """
        return self.model

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

