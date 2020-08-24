"""
Utility functions for Gaussian filtering and smoothing.
"""

from probnum.filtsmooth.statespace import (
    ContinuousModel,
    DiscreteModel,
    LinearSDEModel,
    DiscreteGaussianLinearModel,
)


def is_cont_disc(dynamod, measmod):
    """Checks whether the state space model is continuous-discrete."""
    dyna_is_cont = issubclass(type(dynamod), ContinuousModel)
    meas_is_disc = issubclass(type(measmod), DiscreteModel)
    return dyna_is_cont and meas_is_disc


def is_disc_disc(dynamod, measmod):
    """Checks whether the state space model is discrete-discrete."""
    dyna_is_disc = issubclass(type(dynamod), DiscreteModel)
    meas_is_disc = issubclass(type(measmod), DiscreteModel)
    return dyna_is_disc and meas_is_disc
