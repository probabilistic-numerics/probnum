from probnum.quad.bayesian import *
from probnum.quad.polynomial import *
from probnum.quad.quadrature import *

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
