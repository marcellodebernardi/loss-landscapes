# loss-landscapes

`loss-landscapes` is a library for approximating the value of neural network loss functions in low-dimensional 
subspaces suitable for visualization. The intended purpose of this library is to simplify the production of loss 
landscape visualizations such as those seen in [Visualizing the Loss Landscape of 
Neural Nets](https://arxiv.org/abs/1712.09913v3).

## What is a Loss Landscape?
Let `L : Parameters -> Real Numbers` be a loss function, which maps the vector containing all parameters of a
neural network model to a real number. This is true of loss functions in general, of course, but we focus on
neural networks. Since the network has `n` parameters, the loss function `L` takes an `n`-dimensional input. We
can define the loss landscape as the set of all `n+1`-dimensional points `(param, L(param))`, for all parameter
vectors `param`. For example, the image below, reproduced from the paper mentioned above by Li et al (2018),
provides a visual representation of what a loss function over a two-dimensional parameter space might look like:

![Loss Landscape](/img/loss-landscape.png)

Of course, in reality machine learning models have many more parameters than just two, so the input space of the
loss function is virtually never two-dimensional. In fact, the model whose loss landscape is shown in the image
above actually has far more than two parameters, too. Because we can't produce visualizations in more than three 
dimensions, we cannot hope to visualize the "true" shape of the loss landscape. Instead, a number of techniques
exist for reducing the parameter space to one or two dimensions, ranging from dimensionality reduction techniques
like PCA, to restricting ourselves to a particular subspace of the overall parameter space. For more details,
read Li et al's paper above.

## What does `loss-landscapes` do?
This library facilitates the computation of a neural network model's loss landscape. Its core design principles
are that the library should not mandate any visualization technique or any particular definition of loss, and it
should be agnostic to the numerical computation library in use. As an example, it allows a PyTorch user to examine
the topology of a model's loss function around its current parameters with filter-normalized directions by calling

````python
landscape = loss_landscape.random_plane(model, evaluation_function, normalize='filter')
````

This line would return a 2-dimensional array of loss values, which could in turn be plotted with any software.
Below is a simple contour plot made in `matplotlib` that demonstrates what a planar loss landscape could look like.
Check the `examples` directory for more in-depth examples of what is possible.

![Loss Contour](/img/loss-contour.png)


## Current and Planned Features
Currently, the library has the following features:

1. Computing loss functions along a line defined by two points
2. Computing loss functions along a random direction, starting from a specified point
3. Computing loss functions along a plane defined by three points
4. Computing loss functions along a plane defined by a specified point and two random direction vectors

Furthermore, *filter normalization* can be applied to the random direction vectors. Currently, the only
supported numerical computation library is PyTorch.

Future versions will increase the number of loss landscapes that are computable, as well as expand support
to TensorFlow, scikit-learn, and more.

## Installation
The package will be uploaded on PyPI once it is sufficiently developed. Up to that point, it can be installed
by cloning this repository, and running `pip install -e ./loss-landscapes/`, where `loss-landscapes` is the
directory this repository was cloned into.