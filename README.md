# loss-landscapes

[![CI](https://github.com/marcellodebernardi/loss-landscapes/actions/workflows/ci.yml/badge.svg)](https://github.com/marcellodebernardi/loss-landscapes/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/loss-landscapes.svg)](https://pypi.org/project/loss-landscapes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Project status:** Originally developed in 2019, this project is now back in active development.
> Significant changes to the API and implementation are coming soon.

`loss-landscapes` is a PyTorch library for approximating neural network loss functions, and other related metrics,
in low-dimensional subspaces of the model's parameter space. The library makes the production of visualizations
such as those seen in [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913v3) much
easier, aiding the analysis of the geometry of neural network loss landscapes.

This library does not provide plotting facilities, letting the user define how the data should be plotted.

> **Known limitations:** Some normalisation routines produce incorrect results, gradient evaluation
> has known defects, and the `contrib` modules cannot be imported. See the strict `xfail` tests
> for details before relying on affected functionality.

## Quickstart

Install from [PyPI](https://pypi.org/project/loss-landscapes/):

```bash
python -m pip install loss-landscapes
```

The following example evaluates loss along the line between two small models. It uses untrained
models and synthetic data so you can run it immediately; replace them with your checkpoints and
evaluation data for your own experiments.

```python
import torch

from loss_landscapes import linear_interpolation
from loss_landscapes.metrics import Loss

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
other_model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
inputs = torch.randn(16, 2)
targets = inputs.sum(dim=1, keepdim=True)
metric = Loss(torch.nn.MSELoss(), inputs, targets)

loss_values = linear_interpolation(model, other_model, metric, steps=20, deepcopy_model=True)
print(loss_values.shape)  # (20,)
```

Sampling modifies the supplied model's parameters and disables its parameter gradients by default.
Use `deepcopy_model=True`, as above, to evaluate a copy and preserve the supplied model.

## 1. What is a Loss Landscape?

For a neural network with `n` parameters, let `L : R^n -> R` map its parameter vector to a loss
value. The loss landscape is the graph of this function: the set of points `(θ, L(θ))` in
`R^(n+1)`, for all parameter vectors `θ`.

Since the full parameter space is too high-dimensional to visualise, the loss is usually evaluated
on one- or two-dimensional subspaces. The figure below, reproduced from Li et al., illustrates a
loss function over a two-dimensional parameter space. See
[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913v3) for the
visualisation methods that inspired this library.

<p align="center"><img src="https://raw.githubusercontent.com/marcellodebernardi/loss-landscapes/main/img/loss-landscape.png" width="60%" align="middle"/></p>

## 2. Base Example: Supervised Loss in Parameter Subspaces
A common use case is to evaluate a supervised loss function in a low-dimensional subspace of a
neural network's parameter space. Given a trained `torch.nn.Module` named `model`, a loss function
`loss_function`, inputs `X` and targets `y`, the following evaluates the loss on a random plane:

````python
from loss_landscapes import random_plane
from loss_landscapes.metrics import Loss

metric = Loss(loss_function, X, y)
landscape = random_plane(model, metric, normalization="filter", deepcopy_model=True)
````

The two core concepts are _metrics_ and _parameter subspaces_. The subspace defines the parameter
vectors at which to evaluate the model; the metric defines the quantity to compute at each vector.
Here, `Loss` evaluates the supervised loss over `X` and `y`, and `random_plane` returns a
two-dimensional array of its values. This array can be plotted as a contour map or a surface:

<p align="center"><img src="https://raw.githubusercontent.com/marcellodebernardi/loss-landscapes/main/img/loss-contour.png" width="75%" align="middle"/></p>

<p align="center"><img src="https://raw.githubusercontent.com/marcellodebernardi/loss-landscapes/main/img/loss-contour-3d.png" width="75%" align="middle"/></p>

See [the core-features notebook](examples/core-features.ipynb) for a training and plotting example.
It additionally uses `torchvision`, `matplotlib` and `tqdm`; install these separately.


## 3. Metrics and Custom Metrics
The `loss-landscapes` library can compute any quantity of interest at a collection of points in a parameter subspace,
not just loss. This is accomplished using a `Metric`: a callable object which applies a pre-determined function,
such as a cross entropy loss with a specific set of inputs and outputs, to the model. The `loss_landscapes.metrics`
package provides `Loss`, `LossPerturbations` and `MetricPipeline`. `LossGradient` is also exported,
but currently has the defects noted above.

Custom metrics can be defined by subclassing `Metric` and implementing `__call__`. The library
implementation of `Loss` illustrates this contract:

````python
import torch

from loss_landscapes import ModelWrapper
from loss_landscapes.metrics import Metric


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
For a simple supervised model, functions such as `linear_interpolation` accept a `torch.nn.Module`
directly. More complex cases, such as evaluating a subset of a model's parameters or an RL agent's
expected return, require an explicit interface between the model and the library.

`ModelWrapper` defines this interface. The supplied `GeneralModelWrapper` accepts the model,
the modules whose parameters define the subspace, and a function for forward evaluation.

For example, to vary only the final module of a `torch.nn.Sequential` model while evaluating
the complete model, reuse the loss metric from the supervised example:

```python
from loss_landscapes import GeneralModelWrapper

model_wrapper = GeneralModelWrapper(
    model=model,
    modules=[model[-1]],
    forward_fn=lambda model, inputs: model(inputs),
)
landscape = random_plane(model_wrapper, metric, normalization="filter", deepcopy_model=True)
```

`modules` selects the parameters to vary; the forward function defines how the complete model
is evaluated. This example assumes the final module has parameters to sample.


## 5. Supported Environments

The library supports the following combinations:

| PyTorch minor series | CPython minor versions |
| --- | --- |
| 2.15 | 3.11, 3.12, 3.13, 3.14, 3.15 |

Compatibility is tested on Linux x86-64 CPU with standard, GIL-enabled CPython, using the latest
compatible stable patches in these series. GPU execution, free-threaded builds and alternative
interpreters are not covered by these checks.

See [CONTRIBUTING.md](CONTRIBUTING.md) for compatibility checks and platform prerequisites.

### Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for checkout setup, tests and release instructions, and
[CHANGELOG.md](CHANGELOG.md) for release notes.
