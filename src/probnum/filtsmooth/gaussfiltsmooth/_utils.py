"""
Utility functions for Gaussian filtering and smoothing.
"""

from probnum.filtsmooth.statespace import ContinuousModel, DiscreteModel


def is_cont_disc(dynamod, measmod):
    """Checks whether the state space model is continuous-discrete."""
    dyna_is_cont = isinstance(dynamod, ContinuousModel)
    meas_is_disc = isinstance(measmod, DiscreteModel)
    return dyna_is_cont and meas_is_disc


def is_disc_disc(dynamod, measmod):
    """Checks whether the state space model is discrete-discrete."""
    dyna_is_disc = isinstance(dynamod, DiscreteModel)
    meas_is_disc = isinstance(measmod, DiscreteModel)
    return dyna_is_disc and meas_is_disc
