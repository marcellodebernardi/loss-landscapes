# loss-landscapes

[![CI](https://github.com/marcellodebernardi/loss-landscapes/actions/workflows/ci.yml/badge.svg)](https://github.com/marcellodebernardi/loss-landscapes/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/loss-landscapes.svg)](https://pypi.org/project/loss-landscapes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`loss-landscapes` is a PyTorch library for approximating neural network loss functions, and other related metrics,
in low-dimensional subspaces of the model's parameter space. The library makes the production of visualizations
such as those seen in [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913v3) much
easier, aiding the analysis of the geometry of neural network loss landscapes.

This library does not provide plotting facilities, letting the user define how the data should be plotted.

> **Known limitations:** Some normalisation routines produce incorrect results, gradient evaluation
> has known defects, and the `contrib` modules cannot be imported. See the strict `xfail` tests
> for details before relying on affected functionality.


## 1. What is a Loss Landscape?
Let `L : Parameters -> Real Numbers` be a loss function, which maps a point in the model parameter space to a
real number. For a neural network with `n` parameters, the loss function `L` takes an `n`-dimensional input. We
can define the loss landscape as the set of all `n+1`-dimensional points `(param, L(param))`, for all points
`param` in the parameter space. For example, the image below, reproduced from the paper by Li et al (2018), link
above, provides a visual representation of what a loss function over a two-dimensional parameter space might look
like:

<p align="center"><img src="https://raw.githubusercontent.com/marcellodebernardi/loss-landscapes/main/img/loss-landscape.png" width="60%" align="middle"/></p>

Of course, real machine learning models have a number of parameters much greater than 2, so the parameter space of
the model is virtually never two-dimensional. Because we can't print visualizations in more than two dimensions,
we cannot hope to visualize the "true" shape of the loss landscape. Instead, a number of techniques
exist for reducing the parameter space to one or two dimensions, ranging from dimensionality reduction techniques
like PCA, to restricting ourselves to a particular subspace of the overall parameter space. For more details,
read Li et al's paper.


## 2. Base Example: Supervised Loss in Parameter Subspaces
The simplest use case for `loss-landscapes` is to estimate the value of a supervised loss function in a subspace
of a neural network's parameter space. The subspace in question may be a point, a line, or a plane (these subspaces
can be meaningfully visualized). Suppose the user has trained a supervised learning model, of type `torch.nn.Module`,
on a dataset consisting of samples `X` and labels `y`, by minimizing some loss function. The user now wishes to
produce a surface plot alike to the one in section 1.

This is accomplished as follows:

````python
from loss_landscapes import random_plane
from loss_landscapes.metrics import Loss

metric = Loss(loss_function, X, y)
landscape = random_plane(model, metric, normalization="filter", deepcopy_model=True)
````

As seen in the example above, the two core concepts in `loss-landscapes` are _metrics_ and _parameter subspaces_. The
latter define the section of parameter space to be considered, while the former define what quantity is evaluated at
each considered point in parameter space, and how it is computed. In the example above, we define a `Loss` metric
over data `X` and labels `y`, and instruct `random_plane` to evaluate it in a randomly generated planar subspace.

This would return a 2-dimensional array of loss values, which the user can plot in any desirable way. Example
visualizations the user might use this type of data for are shown below.

<p align="center"><img src="https://raw.githubusercontent.com/marcellodebernardi/loss-landscapes/main/img/loss-contour.png" width="75%" align="middle"/></p>

<p align="center"><img src="https://raw.githubusercontent.com/marcellodebernardi/loss-landscapes/main/img/loss-contour-3d.png" width="75%" align="middle"/></p>

Sampling modifies the supplied model's parameters and disables its parameter gradients by default.
Use `deepcopy_model=True`, as above, to evaluate a copy and preserve the supplied model.
The notebooks in `examples/` use older APIs and need adaptation before use.


## 3. Metrics and Custom Metrics
The `loss-landscapes` library can compute any quantity of interest at a collection of points in a parameter subspace,
not just loss. This is accomplished using a `Metric`: a callable object which applies a pre-determined function,
such as a cross entropy loss with a specific set of inputs and outputs, to the model. The `loss_landscapes.metrics`
package provides `Loss`, `LossPerturbations` and `MetricPipeline`. `LossGradient` is also exported,
but currently has the defects noted above.

Furthermore, the user can add custom metrics by subclassing `Metric`. As an example, consider the library
implementation of `Loss`, for `torch` models:

````python
import abc

import torch

from loss_landscapes import ModelWrapper


class Metric(abc.ABC):
    """A quantity that can be computed given a model or an agent."""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass


class Loss(Metric):
    """Computes a specified loss function over specified input-output pairs."""

    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()
````

Custom metrics receive a `ModelWrapper`. Call `model_wrapper.forward(inputs)` to evaluate
the wrapped model and `model_wrapper.get_modules()` to access the selected PyTorch modules.
A `GeneralModelWrapper` also exposes the underlying model through its `model` attribute.
The wrapper itself is not callable.


## 4. More Complex Models
In the general case of a simple supervised learning model, as in the sections above, client code calls functions
such as `loss_landscapes.linear_interpolation` and passes as argument a PyTorch module of type `torch.nn.Module`.

For more complex cases, such as when the user wants to evaluate the loss landscape as a function of a subset of
the model parameters, or the expected return landscape for a RL agent, the user must specify to the `loss-landscapes`
library how to interface with the model (or the agent, on a more general level). This is accomplished using a
`ModelWrapper` object, which hides the implementation details of the model or agent. For general use, the library
supplies the `GeneralModelWrapper` in the `loss_landscapes.model_interface.model_wrapper` module.

For example, to vary only the final module of a `torch.nn.Sequential` model while evaluating
the complete model, reuse the loss metric from the supervised example:

```python
from loss_landscapes import GeneralModelWrapper

model_wrapper = GeneralModelWrapper(model, [model[-1]], lambda model, inputs: model(inputs))
landscape = random_plane(model_wrapper, metric, normalization="filter", deepcopy_model=True)
```

`modules` selects the parameters to vary; the forward function defines how the complete model
is evaluated. This example assumes the final module has parameters to sample.


## 5. Installation and Use

The library supports the following combinations:

| PyTorch minor series | CPython minor versions |
| --- | --- |
| 2.15 | 3.11, 3.12, 3.13, 3.14, 3.15 |

Compatibility is tested on Linux x86-64 CPU with standard, GIL-enabled CPython, using the latest
compatible stable patches in these series. GPU execution, free-threaded builds and alternative
interpreters are not covered by these checks.

Install from [PyPI](https://pypi.org/project/loss-landscapes/):

```bash
python -m pip install loss-landscapes
```

This installs NumPy and PyTorch automatically. Import the library as follows:

```python
import loss_landscapes
import loss_landscapes.metrics
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for compatibility checks and platform prerequisites.

### Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for checkout setup, tests and release instructions, and
[CHANGELOG.md](CHANGELOG.md) for release notes.
