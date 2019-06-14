import abc
import loss_landscapes.model_interface.model_tensor as model_tensor


class ModelWrapper(abc.ABC):
    def __init__(self, model, components, layers, call_fn):
        self.model = model                    # wrapped model
        self.components = components          # how to get state
        self.layers = layers                  # layers to include
        self.forward_fn = call_fn             # how to use model in evaluation

    def get_model(self):
        """
        Return a reference to the model encapsulated in this wrapper.
        :return: wrapped model
        """
        return self.model

    @abc.abstractmethod
    def __call__(self, x):
        """
        Calls the model or agent on the given inputs, and returns the desired output.
        :param x: inputs to the model or agent
        :return: model output
        """
        pass

    @abc.abstractmethod
    def get_parameter_tensor(self, deepcopy=False) -> model_tensor.ParameterTensor:
        """
        Return a ParameterTensor wrapping the parameters of the underlying model. The
        parameters can either be returned as a view of the model parameters or as a copy.
        :param deepcopy: whether to view or deepcopy the model parameters
        :return: view or deepcopy of accessible model parameters
        """
        pass

    @abc.abstractmethod
    def set_parameter_tensor(self, new_parameters: model_tensor.ParameterTensor):
        """
        Sets the parameters of the wrapped model to the given ParameterVector.
        :param new_parameters: ParameterVector of new parameters
        :return: none
        """
        pass

