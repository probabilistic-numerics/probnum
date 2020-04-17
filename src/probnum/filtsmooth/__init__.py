"""
Bayesian filtering and smoothing.

Bayesian quadrature methods integrate a function by iteratively building a probabilistic model and using its predictions
to adaptively choose points to evaluate the integrand.
"""

from .statespace import *
from .bayesianfilter import *
from .gaussfiltsmooth import *


# Public classes and functions. Order is reflected in documentation.
__all__ = ["BayesianFilter", "GaussianSmoother", "GaussianFilter",
           "KalmanFilter", "KalmanSmoother",
           "ExtendedKalmanFilter", "ExtendedKalmanSmoother",
           "UnscentedKalmanFilter", "UnscentedKalmanSmoother",
           "UnscentedTransform",
           "ContinuousModel", "LinearSDEModel", "LTISDEModel",
           "DiscreteModel", "DiscreteGaussianModel",
           "DiscreteGaussianLinearModel", "DiscreteGaussianLTIModel",
           "generate_cd", "generate_dd"]
