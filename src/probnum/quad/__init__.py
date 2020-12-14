"""Quadrature / Numerical Integration of Functions.

This package implements Bayesian quadrature rules used for numerical
integration of functions on a given domain. Such methods integrate a
function by iteratively building a probabilistic model and adaptively
choosing points to evaluate the integrand based on said model.
"""

from ._bayesquad import *
from .bq_methods import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "bayesquad",
    "BayesianQuadrature",
    "WarpedBayesianQuadrature",
]

# Set correct module paths. Corrects links and module paths in documentation.
BayesianQuadrature.__module__ = "probnum.quad"
WarpedBayesianQuadrature.__module__ = "probnum.quad"
