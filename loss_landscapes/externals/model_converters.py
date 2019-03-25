import abc
import torch
import torch.nn
from state_representation import LayeredVector


class ModelConverter(abc.ABC):
    """
    Base class for objects which convert between the representation of model parameters
    used by the external library in use (torch, tensorflow, etc), and the internal numpy
    representation.
    """

    @abc.abstractmethod
    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def to_internal(self, data):
        pass

    @abc.abstractmethod
    def to_external(self, data):
        pass

    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def subtract(self, data):
        pass

    def model(self):
        return self.model


class Pytorch(ModelConverter):
    """ Converter for parameter lists obtained from PyTorch. """
    def __init__(self, model):
        super().__init__(model)
        for p in model.parameters():
            p.requires_grad = False

    def to_internal(self, torch_parameters) -> LayeredVector:
        """ Produces a LayeredVector to represent the torch parameters. """
        assert _is_torch(torch_parameters)

        parameters = LayeredVector()
        for p in torch_parameters:
            parameters.add_layer(p.clone().detach().numpy())
        return parameters

    def to_external(self, numpy_parameters: LayeredVector) -> list:
        """ Converts the given LayeredVector of numpy arrays to a list of torch parameters. """
        assert _is_layered_vector(numpy_parameters)

        parameters = []
        for p in numpy_parameters:
            parameters.append(torch.nn.Parameter(torch.from_numpy(p), requires_grad=False))
        return parameters

    def add(self, numpy_parameters: LayeredVector):
        numpy_parameters = self.to_external(numpy_parameters)

        for p_idx, model_p in enumerate(self.model.parameters(), 0):
            model_p += numpy_parameters[p_idx]

    def subtract(self, numpy_parameters: LayeredVector):
        numpy_parameters = self.to_external(numpy_parameters)

        for p_idx, model_p in enumerate(self.model.parameters(), 0):
            model_p -= numpy_parameters[p_idx]


class TensorFlow(ModelConverter):
    def __init__(self, model):
        super().__init__(model)
        pass

    def to_internal(self, tf_weights) -> LayeredVector:
        raise NotImplementedError()

    def to_external(self, numpy_weights):
        raise NotImplementedError()

    def add(self, numpy_weights: LayeredVector):
        pass

    def subtract(self, numpy_weights: LayeredVector):
        pass


def build_converter(model) -> ModelConverter:
    if isinstance(model, torch.nn.Module):
        return Pytorch(model)
    else:
        raise ValueError('Cannot build converter for model of type ' + str(type(model)))


def _is_torch(data) -> bool:
    """ Returns true if the given data is interpretable as the internal state of a torch.nn neural network. """
    try:
        assert isinstance(data, list)
        return isinstance(data[0], torch.nn.Parameter)
    except AssertionError:
        return False


def _is_layered_vector(data) -> bool:
    """ Returns true if the given data is a LayeredVector """
    return isinstance(data, LayeredVector)
