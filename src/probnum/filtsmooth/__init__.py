"""Bayesian filtering and smoothing."""

from .bayesfiltsmooth import *
from .filtsmoothposterior import FiltSmoothPosterior
from .gaussfiltsmooth import *
from .statespace import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Kalman",
    "ContinuousEKFComponent",
    "ContinuousUKFComponent",
    "DiscreteEKFComponent",
    "DiscreteUKFComponent",
    "UnscentedTransform",
    "Transition",
    "SDE",
    "LinearSDE",
    "LTISDE",
    "DiscreteGaussian",
    "DiscreteLinearGaussian",
    "DiscreteLTIGaussian",
    "FiltSmoothPosterior",
    "KalmanPosterior",
    "generate_cd",
    "generate_dd",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
Transition.__module__ = "probnum.filtsmooth.statespace"
SDE.__module__ = "probnum.filtsmooth.statespace"
LinearSDE.__module__ = "probnum.filtsmooth.statespace"
DiscreteGaussian.__module__ = "probnum.filtsmooth.statespace"
DiscreteLinearGaussian.__module__ = "probnum.filtsmooth.statespace"
KalmanPosterior.__module__ = "probnum.filtsmooth"
