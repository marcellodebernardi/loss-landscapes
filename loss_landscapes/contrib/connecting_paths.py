"""
This module exposes functions for loss landscape operations which are more complex than simply
computing the loss at different points in parameter space. This includes things such as Kolsbjerg
et al.'s Automated Nudged Elastic Band algorithm.
"""


import abc
import copy
import numpy as np
from loss_landscapes.model_interface.model_interface import wrap_model


class _ParametricCurve(abc.ABC):
    """ A _ParametricCurve is used in the Garipov path search algorithm. """
    # todo


class _PolygonChain(_ParametricCurve):
    """ A _ParametricCurve consisting of consecutive line segments. """
    # todo
    pass


class _BezierCurve(_ParametricCurve):
    """
    A Bezier curve is a parametric curve defined by a set of control points, including
    a start point and an end-point. The order of the curve refers to the number of control
    points excluding the start point: for example, an order 1 (linear) Bezier curve is
    defined by 2 points, an order 2 (quadratic) Bezier curve is defined by 3 points, and
    so on.

    In this library, each point is a neural network model with a specific value assignment
    to the model parameters.
    """
    def __init__(self, model_start, model_end, order=2):
        """
        Define a Bezier curve between a start point and an end point. The order of the
        curve refers to the number of control points, excluding the start point. The default
        order of 1, for example, results in no further control points being added after
        the given start and end points.

        :param model_start: point defining start of curve
        :param model_end: point defining end of curve
        :param order: number of control points, excluding start point
        """
        super().__init__()
        if order != 2:
            raise NotImplementedError('Currently only order 2 bezier curves are supported.')

        self.model_start_wrapper = wrap_model(copy.deepcopy(model_start))
        self.model_end_wrapper = wrap_model(copy.deepcopy(model_end))
        self.order = order
        self.control_points = []

        # add intermediate control points
        if order > 1:
            start_parameters = self.model_start_wrapper.get_parameter_tensor()
            end_parameters = self.model_end_wrapper.get_parameter_tensor()
            direction = (end_parameters - start_parameters) / order

            for i in range(1, order):
                model_template_wrapper = copy.deepcopy(self.model_start_wrapper)
                model_template_wrapper.set_parameter_tensor(start_parameters + (direction * i))
                self.control_points.append(model_template_wrapper)

    def fit(self):
        # todo
        raise NotImplementedError()


def auto_neb() -> np.ndarray:
    """ Automatic Nudged Elastic Band algorithm, as used in https://arxiv.org/abs/1803.00885 """
    # todo return list of points in parameter space to represent trajectory
    # todo figure out how to return points as coordinates in 2D
    raise NotImplementedError()


def garipov_curve_search(model_a, model_b, curve_type='polygon_chain') -> np.ndarray:
    """
    We refer by 'Garipov curve search' to the algorithm proposed by Garipov et al (2018) for
    finding low-loss paths between two arbitrary minima in a loss landscape. The core idea
    of the method is to define a parametric curve in the model's parameter space connecting
    one minima to the other, and then minimizing the expected loss along this curve by
    modifying its parameterization. For details, see https://arxiv.org/abs/1802.10026

    This is an alternative to the auto_neb algorithm.
    """
    model_a_wrapper = wrap_model(model_a)
    model_b_wrapper = wrap_model(model_b)

    point_a = model_a_wrapper.get_parameter_tensor()
    point_b = model_b_wrapper.get_parameter_tensor()

    # todo
    if curve_type == 'polygon_chain':
        raise NotImplementedError('Not implemented yet.')
    elif curve_type == 'bezier_curve':
        raise NotImplementedError('Not implemented yet.')
    else:
        raise AttributeError('Curve type is not polygon_chain or bezier_curve.')
