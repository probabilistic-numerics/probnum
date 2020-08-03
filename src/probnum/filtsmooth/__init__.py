"""
Bayesian filtering and smoothing.
"""

from .statespace import *
from .gaussfiltsmooth import *
from .bayesfiltsmooth import *
from .filtsmoothposterior import FiltSmoothPosterior

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "GaussFiltSmooth",
    "Kalman",
    "ExtendedKalman",
    "UnscentedKalman",
    "UnscentedTransform",
    "ContinuousModel",
    "LinearSDEModel",
    "LTISDEModel",
    "DiscreteModel",
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
ContinuousModel.__module__ = "probnum.filtsmooth"
LinearSDEModel.__module__ = "probnum.filtsmooth"
DiscreteModel.__module__ = "probnum.filtsmooth"
DiscreteGaussianModel.__module__ = "probnum.filtsmooth"
DiscreteGaussianLinearModel.__module__ = "probnum.filtsmooth"
GaussFiltSmooth.__module__ = "probnum.filtsmooth"
KalmanPosterior.__module__ = "probnum.filtsmooth"
