# loss-landscapes

`loss-landscapes` is a library for approximating neural network loss functions, and other related metrics, 
along low-dimensional subspaces of the model parameter space. The library makes the production of visualizations
such as those seen in [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913v3) much
easier, aiding the analysis of the geometry of neural network loss landscapes.

Currently, `loss-landscapes` only supports PyTorch models, but support for other DL libraries (TensorFlow in particular)
is planned for future releases.


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
evaluator = LossEvaluator(loss_function, X, y)
landscape = random_plane(model, evaluator, normalize='filter')
````

This would return a 2-dimensional array of loss values, which the user can plot in any desirable way. 
Below are a contour plot and a surface plot made in `matplotlib`, drawn over the same loss landscape data, that 
demonstrate what a loss landscape along a planar parameter subspace could look like.
Check the `examples` directory for `jupyter` notebooks with more in-depth examples of what is possible.

<p align="center"><img src="/img/loss-contour.png" width="75%" align="middle"/></p>

<p align="center"><img src="/img/loss-contour-3d.png" width="75%" align="middle"/></p>


## 3. Evaluators and Custom Evaluators
The `loss-landscapes` library can compute any quantity of interest at a collection of points in a parameter subspace,
not just loss. This is accomplished using an `Evaluator`: a callable object which applies a pre-determined function,
such as a cross entropy loss with a specific set of inputs and outputs, at every point. The `loss_landscapes.evaluators`
package contains a number of evaluators that cover common use cases, such as `LossEvaluator` (evaluates a loss
function), `GradientEvaluator` (evaluates the gradient of the loss w.r.t. the model parameters), 
`PrincipalCurvatureEvaluator` (evaluates the principal curvatures of the loss function), and more.

Furthermore, the user can add custom evaluators by subclassing `Evaluator`. As an example, consider the library
implementation of `LossEvaluator`, for `torch` models:

````python
class LossEvaluator(SupervisedTorchEvaluator):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, supervised_loss_fn, inputs, target):
        super().__init__(supervised_loss_fn, inputs, target)

    def __call__(self, model) -> np.ndarray:
        return self.loss_fn(model(self.inputs), self.target).clone().detach().numpy()
````

In summary, the `Evaluator` abstraction adds a great degree of flexibility. An evaluator defines what quantity
dependent on model parameters the user is interested in evaluating , and how to evaluate it. The user could define, 
for example, an evaluator that computes an estimate of the expected return of a reinforcement learning agent.


## 4. RL Agents and Other Complications
In the general case of a simple supervised learning model, as in the sections above, client code calls functions 
such as `loss_landscapes.linear_interpolation` and passes as argument a reference to a deep learning model. The 
`loss-landscapes` library detects the DL library to which the model belongs. This process is not visible to the
user and in most cases can safely be ignored.

For more complex cases, such as when the user wants to evaluate the loss landscape as a function of a subset of
the model parameters, or the expected return landscape for a RL model, the user must specify to the `loss-landscapes`
library how to interface with the model (or the agent, on a more general level). This is accomplished using a
`AgentInterface` object. In the example below, the `AgentInterface` specifies the means by which the `random_plane`
method will interface with a particular reinforcement learning agent, such that the agent object contains neural
network models.

````python
# agent.policy and agent.value_function are pytorch modules
interface = AgentInterface(library='torch', components=[agent.policy, agent.value_function], call_fn= lambda x: model.policy(x))
landscape = random_plane(agent, evaluator, normalize='filter')
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
import loss_landscapes.evaluators  # for the base Evaluator class
import loss_landscapes.evaluators.torch  # for the pre-defined PyTorch evaluators
````