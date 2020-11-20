"""Bayesian Quadrature."""

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
