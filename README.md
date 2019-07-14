# loss-landscapes

`loss-landscapes` is a library for approximating neural network loss functions, and other related metrics, 
along low-dimensional subspaces of the model parameter space. The library makes the production of visualizations
such as those seen in [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913v3) much
easier, aiding the analysis of the geometry of neural network loss landscapes.

Currently, `loss-landscapes` only supports PyTorch models, but support for other DL libraries (TensorFlow in particular)
is planned for future releases.

**NOTE: this library is in early development. Bugs are virtually a certainty, and the API is volatile. Do not use
this library in production code. For prototyping and research, always use the newest version of the library.**


## 1. What is a Loss Landscape?
Let `L : Parameters -> Real Numbers` be a loss function, which maps a point in the model parameter space to a 
real number. For a neural network with `n` parameters, the loss function `L` takes an `n`-dimensional input. We
can define the loss landscape as the set of all `n+1`-dimensional points `(param, L(param))`, for all points
`param` in the parameter space. For example, the image below, reproduced from the paper by Li et al (2018), link
above, provides a visual representation of what a loss function over a two-dimensional parameter space might look 
like:

<p align="center"><img src="/img/loss-landscape.png" width="60%" align="middle"/></p>

Of course, real machine learning models have a number of parameters much greater than 2, so the parameter space of 
the model is virtually never two-dimensional. Because we can't print visualizations in more than two dimensions, 
we cannot hope to visualize the "true" shape of the loss landscape. Instead, a number of techniques
exist for reducing the parameter space to one or two dimensions, ranging from dimensionality reduction techniques
like PCA, to restricting ourselves to a particular subspace of the overall parameter space. For more details,
read Li et al's paper.


## 2. Loss in Parameter Subspaces
This library facilitates the computation of a neural network model's loss landscape in low-dimensional subspaces
of the parameter space. It does not provide plotting facilities, letting the user define how the data should be plotted,
and is designed to support any deep learning library (in principle - currently only PyTorch is supported). As an 
example, it allows a PyTorch user to produce data for a plot such as the one seen above by simply calling

````python
metric = Loss(loss_function, X, y)
landscape = random_plane(model, metric, normalize='filter')
````

This would return a 2-dimensional array of loss values, which the user can plot in any desirable way. 
Below are a contour plot and a surface plot made in `matplotlib`, drawn over the same loss landscape data, that 
demonstrate what a loss landscape along a planar parameter subspace could look like.
Check the `examples` directory for `jupyter` notebooks with more in-depth examples of what is possible.

<p align="center"><img src="/img/loss-contour.png" width="75%" align="middle"/></p>

<p align="center"><img src="/img/loss-contour-3d.png" width="75%" align="middle"/></p>


## 3. Metrics and Custom Metrics
The `loss-landscapes` library can compute any quantity of interest at a collection of points in a parameter subspace,
not just loss. This is accomplished using a `Metric`: a callable object which applies a pre-determined function,
such as a cross entropy loss with a specific set of inputs and outputs, to the model. The `loss_landscapes.model_metrics`
package contains a number of metrics that cover common use cases, such as `Loss` (evaluates a loss
function), `LossGradient` (evaluates the gradient of the loss w.r.t. the model parameters), 
`PrincipalCurvatureEvaluator` (evaluates the principal curvatures of the loss function), and more.

Furthermore, the user can add custom metrics by subclassing `Metric`. As an example, consider the library
implementation of `Loss`, for `torch` models:

````python
class Metric(abc.ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass


class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: TorchModelWrapper) -> np.ndarray:
        return self.loss_fn(model_wrapper(self.inputs), self.target).clone().detach().numpy()
````

The user may create custom `Metric`s in a similar manner. One complication is that the `Metric` class' 
`__call__` method is designed to take as input a `ModelWrapper` rather than a model. This class is internal
to the library and exists to facilitate the handling of the myriad of different models a user may pass as
inputs to a function such as `loss_landscapes.planar_interpolation()`. It is sufficient for the user to know
that a `ModelWrapper` is a callable object that can be used to call the model on a given input (see the `call_fn`
argument of the `ModelInterface` class in the next section). The class also provides a `get_model()` method
that exposes a reference to the underlying model, should the user wish to carry out more complicated operations
on it.

In summary, the `Metric` abstraction adds a great degree of flexibility. An metric defines what quantity
dependent on model parameters the user is interested in evaluating, and how to evaluate it. The user could define, 
for example, a metric that computes an estimate of the expected return of a reinforcement learning agent.


## 4. RL Agents and Other Complications
In the general case of a simple supervised learning model, as in the sections above, client code calls functions 
such as `loss_landscapes.linear_interpolation` and passes as argument a reference to a deep learning model. The 
`loss-landscapes` library detects the DL library to which the model belongs. This process is not visible to the
user and in most cases can safely be ignored.

For more complex cases, such as when the user wants to evaluate the loss landscape as a function of a subset of
the model parameters, or the expected return landscape for a RL model, the user must specify to the `loss-landscapes`
library how to interface with the model (or the agent, on a more general level). This is accomplished using a
`ModelInterface` object. In the example below, the `ModelInterface` specifies the means by which the `random_plane`
method will interface with a particular reinforcement learning agent, such that the agent object contains neural
network models.

In the example below, a RL agent with a policy network and a value function network is being evaluated on some
metric.

````python
# agent.policy and agent.value_function are pytorch modules
interface = ModelInterface(library='torch', components=[agent.policy, agent.value_function], call_fn= lambda x: model.policy(x))
landscape = random_plane(agent, metric, interface, normalize='filter')
````



## 5. WIP: Connecting Paths, Saddle Points, and Trajectory Tracking
A number of features are currently under development, but as of yet incomplete.

A number of papers in recent years have shown that loss landscapes of neural networks are dominated by a
proliferation of saddle points, that good solutions are better described as large low-loss plateaus than as
"well-bottom" points, and that for sufficiently high-dimensional networks, a low-loss path in parameter space can
be found between almost any arbitrary pair of minima. In the future, the `loss-landscapes` library will feature 
implementations of algorithms for finding such low-loss connecting paths in the loss landscape, as well as tools to
facilitate the study of saddle points.

Some sort of trajectory tracking features are also under consideration, though at the time it's unclear what this
should actually mean, as the optimization trajectory is implicitly tracked by the user's training loop. Any metric
along the optimization trajectory can be tracked with libraries such as [ignite](https://github.com/pytorch/ignite)
for PyTorch.


## 7. Support for Other DL Libraries
Once the currently envisioned features are complete, the first priority will be adding support for TensorFlow.


## 8. Installation and Use
The package is available on PyPI. Install using `pip install loss-landscapes`. To use the library, import as follows:

````python
import loss_landscapes
import loss_landscapes.model_metrics        # for the base Metric class
import loss_landscapes.model_metrics.torch  # for the pre-defined PyTorch metrics, split into sl_metrics and rl_metrics
````