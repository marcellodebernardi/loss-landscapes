import torch
import copy


def linear_interpolation(model, start_params, end_params, evaluation_function, steps=100):
    """
    Returns an approximation of the loss of the model along a linear subspace of the
    parameter space defined by two end points.

    That is, given a set of parameters 'start_params' and 'end_params', both of which
    define a point in parameter space, the loss is computed at N points along the
    straight line connecting the two points. A common choice is to use the weights
    before training and the weights after convergence as the start and end points of
    the line.

    Note that a simple linear interpolation can produce misleading approximations
    of the loss landscape due to the scale invariance of neural networks. The sharpness/
    flatness of minima or maxima is affected by the scale of the neural network weights.
    For more details, see `https://arxiv.org/abs/1712.09913v3`. It is recommended to
    use the filter normalization option.

    The evaluation function supplied has to be of the form

        evaluation_function(model: torch.nn.Module)

    and must specify a procedure whereby the model passed to it is evaluated on the
    task of interest, returning the resulting loss.

    Args:
        model: torch.nn.Module model to visualize
        start_params: the state_dict of the model at the start point of the line
        end_params: the state_dict of the model at the end point of the line
        evaluation_function: an evaluation function that provides a loss for the model
        steps: the number of points along the line for which loss is computed
    Returns:
        list: the losses for each step along the linear interpolation
    """
    # check that model, start_params and end_params are compatible
    # todo

    # todo filter normalization

    # copy model and parameters to avoid aliasing effects
    model = copy.deepcopy(model)
    start_params = copy.deepcopy(start_params)
    end_params = copy.deepcopy(end_params)

    # compute the direction vector between start and end points, then scale by step size
    direction_dict = dict()
    alpha = 1 / steps
    for layer_name in start_params.keys():
        direction_dict[layer_name] = alpha * (end_params[layer_name] - start_params[layer_name])

    # set model to start parameters
    for layer_name in model.state_dict().keys():
        model.state_dict()[layer_name] = start_params[layer_name]

    # compute losses
    losses = []
    for step in range(steps):
        for layer_name in model.state_dict().keys():
            model.state_dict()[layer_name] += direction_dict[layer_name]

        losses.append(evaluation_function(model))

    return losses