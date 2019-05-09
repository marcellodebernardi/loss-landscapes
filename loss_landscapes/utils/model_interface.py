"""
Defines functions for getting and setting the parameters of a model.
"""

import copy
from loss_landscapes.utils.parameter_vector import ParameterVector

SUPPORTED = {
    'torch.nn.Module': 'torch',
    'tensorflow.keras.Model': 'tf_keras'
}


def _import_torch():
    import torch as t
    return t


def _import_dl_library(model_type):
    if model_type == 'torch':
        return _import_torch()


def _detect_model_type(model) -> str:
    """
    Returns a string indicating the type and library of
    :param model: the model of which we want to know the type
    :return: a string indicating the deep learning library the model belongs to
    """
    model_type = str(type(model))
    try:
        return SUPPORTED[model_type]
    except KeyError:
        _raise_unrecognized_model_error(model_type)


def _raise_unrecognized_model_error(model_type: str):
    """
    Raises an error informing the user that the deep learning library they are using is
    not supported by loss_landscapes.
    :return: none
    """
    # this is its own method because the raise expression below is used in multiple places
    raise ValueError(
        'Model type not recognized. Model is of type ' + str(model_type) + ', supported include ' + str(
            SUPPORTED)
    )


class ModelInterface:
    """ Wraps a model and defines the operations for getting its weights or updating them. """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_type = _detect_model_type(model)
        self.library = _import_dl_library(self.model_type)

        # torch-specific state
        # torch_keys is used to maintain a consistent order of the internal numpy array lists
        self.torch_keys = list(self.model.state_dict().keys()) if self.model_type == 'torch' else None

    def get_model(self):
        """
        Returns a reference to the model wrapped by this ModelInterface.
        :return: wrapped model
        """
        return self.model

    def build_parameter_vector(self) -> ParameterVector:
        """
        Returns the parameters of the model as a list of numpy arrays.
        :return: list of numpy arrays
        """
        #
        if self.model_type == 'torch':
            parameters = []
            # use keys from stored key list to ensure list is consistently ordered
            for key in self.torch_keys:
                parameters.append(copy.deepcopy(self.model.state_dict()[key].numpy()))
            return ParameterVector(parameters)

        elif self.model_type == 'tf_keras':
            # tf and keras support coming in the future, but not yet implemented
            _raise_unrecognized_model_error(self.model_type)

        else:
            _raise_unrecognized_model_error(self.model_type)

    def set_parameters(self, new_parameters: ParameterVector):
        """
        Sets the model's parameters using the given list of parameters.
        :param new_parameters: list of numpy arrays
        :return: none
        """
        if self.model_type == 'torch':
            new_state_dict = dict()
            for i in range(len(new_parameters)):
                new_state_dict[self.torch_keys[i]] = self.library.from_numpy(copy.deepcopy(new_parameters[i]))
            self.model.load_dict(self.torch_keys)
