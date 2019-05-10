"""
This module exposes functions for loss landscape operations which are more complex than simply
computing the loss at different points in parameter space. This includes things such as Kolsbjerg
et al.'s Automated Nudged Elastic Band algorithm.
"""


import numpy as np


def auto_neb() -> np.ndarray:
    """ Automatic Nudged Elastic Band algorithm, as used in https://arxiv.org/abs/1803.00885 """
    # todo return list of points in parameter space to represent trajectory
    # todo figure out how to return points as coordinates in 2D
    raise NotImplementedError()


def garipov_method() -> np.ndarray:
    """ Similar to auto_neb, but using algorithm from https://arxiv.org/abs/1802.10026"""
    raise NotImplementedError()


def find_saddle_point() -> np.ndarray:
    """ Gradient traversal to find saddle points, inspired by https://arxiv.org/abs/1406.2572 """
    raise NotImplementedError()



