"""
Bayesian filtering and smoothing.

Bayesian quadrature methods integrate a function by iteratively building a probabilistic model and using its predictions
to adaptively choose points to evaluate the integrand.
"""

from .statespace import *
from .gaussfiltsmooth import *

# Public classes and functions. Order is reflected in documentation.
__all__ = ["GaussianSmoother", "GaussianFilter",
           "KalmanFilter", "KalmanSmoother",
           "ExtendedKalmanFilter", "ExtendedKalmanSmoother",
           "UnscentedKalmanFilter", "UnscentedKalmanSmoother",
           "UnscentedTransform",
           "ContinuousModel", "LinearSDEModel", "LTISDEModel",
           "DiscreteModel", "DiscreteGaussianModel",
           "DiscreteGaussianLinearModel", "DiscreteGaussianLTIModel",
           "generate_cd", "generate_dd"]

# Set correct module paths (for superclasses). Corrects links and module paths in documentation.
ContinuousModel.__module__ = "probnum.filtsmooth"
LinearSDEModel.__module__ = "probnum.filtsmooth"
DiscreteModel.__module__ = "probnum.filtsmooth"
DiscreteGaussianModel.__module__ = "probnum.filtsmooth"
DiscreteGaussianLinearModel.__module__ = "probnum.filtsmooth"
GaussianFilter.__module__ = "probnum.filtsmooth"
GaussianSmoother.__module__ = "probnum.filtsmooth"
