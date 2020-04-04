"""
(Bayesian) Quadrature.

Bayesian quadrature methods integrate a function by iteratively building a probabilistic model and using its predictions
to adaptively choose points to evaluate the integrand.
"""

from .bayesquadrature import *

# Public classes and functions. Order is reflected in documentation.
__all__ = ["bayesquad", "nbayesquad", "BayesianQuadrature", "VanillaBayesianQuadrature", "WASABIBayesianQuadrature"]
