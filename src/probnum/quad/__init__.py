"""
Quadrature, i.e. numerical integration.

This module collects both classic and Bayesian quadrature rules used for numerical
integration of functions.

Bayesian quadrature methods integrate a function by iteratively building a probabilistic
model and using its predictions to adaptively choose points to evaluate the integrand.
"""

from probnum.quad.quadrature import *
from probnum.quad.bayesian import *
from probnum.quad.polynomial import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "quad",
    "nquad",
    "bayesquad",
    "nbayesquad",
    "Quadrature",
    "PolynomialQuadrature",
    "BayesianQuadrature",
    "VanillaBayesianQuadrature",
    "WSABIBayesianQuadrature",
    "ClenshawCurtis",
]

# Set correct module paths. Corrects links and module paths in documentation.
Quadrature.__module__ = "probnum.quad"
BayesianQuadrature.__module__ = "probnum.quad"
PolynomialQuadrature.__module__ = "probnum.quad"
