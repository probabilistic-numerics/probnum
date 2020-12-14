"""Bayesian Filtering and Smoothing.

This package provides different kinds of Bayesian filters and smoothers
which estimate the distribution over observed and hidden variables in a
sequential model. The two operations differ by what information they
use. Filtering considers all observations up to a given point, while
smoothing takes the entire set of observations into account.
"""

from .bayesfiltsmooth import *
from .filtsmoothposterior import *
from .gaussfiltsmooth import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Kalman",
    "ContinuousEKFComponent",
    "DiscreteEKFComponent",
    "ContinuousUKFComponent",
    "DiscreteUKFComponent",
    "UnscentedTransform",
    "FiltSmoothPosterior",
    "KalmanPosterior",
    "IteratedKalman",
    "StoppingCriterion",
    "FixedPointStopping",
]
