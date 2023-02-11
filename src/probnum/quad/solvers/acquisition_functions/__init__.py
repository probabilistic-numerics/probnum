"""Acquisition functions for Bayesian quadrature."""

from ._acquisition_function import AcquisitionFunction
from ._integral_variance_reduction import IntegralVarianceReduction
from ._mutual_information import MutualInformation
from ._predictive_variance import WeightedPredictiveVariance

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "AcquisitionFunction",
    "IntegralVarianceReduction",
    "MutualInformation",
    "WeightedPredictiveVariance",
]

# Set correct module paths. Corrects links and module paths in documentation.
AcquisitionFunction.__module__ = "probnum.quad.solvers.acquisition_functions"
IntegralVarianceReduction.__module__ = "probnum.quad.solvers.acquisition_functions"
MutualInformation.__module__ = "probnum.quad.solvers.acquisition_functions"
WeightedPredictiveVariance.__module__ = "probnum.quad.solvers.acquisition_functions"
