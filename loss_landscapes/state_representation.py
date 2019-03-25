# the state representation defined in this module exists mostly to clarify
# the semantics of some of the code; the meaning of certain operations is
# clearer if defined on a "state vector" rather than on a generic list, even
# though the state vector in turn just wraps a list.

import numpy as np


class LayeredVector:
    def __init__(self, layers: list = None):
        self.layers = []
        if layers is not None:
            for l in layers:
                assert isinstance(l, np.ndarray)
                self.layers.append(l)

    def __getitem__(self, index):
        return self.layers[index]

    def __setitem__(self, key, value):
        self.layers[key] = value

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def layer(self, index):
        return self.layers[index]

    def add_layer(self, layer):
        assert isinstance(layer[0], np.ndarray)
        self.layers.append(layer)
