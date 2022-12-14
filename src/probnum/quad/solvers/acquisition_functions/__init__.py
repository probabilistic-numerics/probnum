"""Acquisition functions for Bayesian quadrature."""

from ._acquisition_function import AcquisitionFunction
from ._predictive_variance import WeightedPredictiveVariance

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "AcquisitionFunction",
    "WeightedPredictiveVariance",
]

# Set correct module paths. Corrects links and module paths in documentation.
WeightedPredictiveVariance.__module__ = "probnum.quad.solvers.acquisition_functions"
