"""Bayesian filtering and smoothing."""

from .bayesfiltsmooth import *
from .filtsmoothposterior import FiltSmoothPosterior
from .gaussfiltsmooth import *
from .statespace import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Kalman",
    "ContinuousEKF",
    "ContinuousUKF",
    "DiscreteEKF",
    "DiscreteUKF",
    "UnscentedTransform",
    "Transition",
    "SDE",
    "LinearSDE",
    "LTISDE",
    "DiscreteGaussianModel",
    "DiscreteGaussianLinearModel",
    "DiscreteGaussianLTIModel",
    "FiltSmoothPosterior",
    "KalmanPosterior",
    "generate_cd",
    "generate_dd",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
Transition.__module__ = "probnum.filtsmooth"
SDE.__module__ = "probnum.filtsmooth"
LinearSDE.__module__ = "probnum.filtsmooth"
DiscreteGaussianModel.__module__ = "probnum.filtsmooth"
DiscreteGaussianLinearModel.__module__ = "probnum.filtsmooth"
KalmanPosterior.__module__ = "probnum.filtsmooth"
