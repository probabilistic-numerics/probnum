"""
Quadrature, i.e. numerical integration.

This module collects both classic and Bayesian quadrature rules used for numerical
integration of functions.

Bayesian quadrature methods integrate a function by iteratively building a probabilistic
model and using its predictions to adaptively choose points to evaluate the integrand.
"""

from probnum.quad.bayesian_quadrature import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "bayesquad",
    "BayesianQuadrature",
    "VanillaBayesianQuadrature",
    "WSABIBayesianQuadrature",
]

# Set correct module paths. Corrects links and module paths in documentation.
BayesianQuadrature.__module__ = "probnum.quad"
