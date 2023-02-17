"""Covariance functions.

Covariance functions describe the spatial or temporal variation of a random process.
If evaluated at two sets of points, a covariance function computes the covariance of the
values of the random process at these locations.

Covariance functions support basic algebraic operations, including scaling, addition
and multiplication.
"""

from ._covariance_function import CovarianceFunction, IsotropicMixin
from ._exponentiated_quadratic import ExpQuad
from ._linear import Linear
from ._matern import Matern
from ._polynomial import Polynomial
from ._product_matern import ProductMatern
from ._rational_quadratic import RatQuad
from ._white_noise import WhiteNoise

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "CovarianceFunction",
    "IsotropicMixin",
    "WhiteNoise",
    "Linear",
    "Polynomial",
    "ExpQuad",
    "RatQuad",
    "Matern",
    "ProductMatern",
]

# Set correct module paths. Corrects links and module paths in documentation.
CovarianceFunction.__module__ = "probnum.randprocs.covfuncs"
IsotropicMixin.__module__ = "probnum.randprocs.covfuncs"

WhiteNoise.__module__ = "probnum.randprocs.covfuncs"
Linear.__module__ = "probnum.randprocs.covfuncs"
Polynomial.__module__ = "probnum.randprocs.covfuncs"
ExpQuad.__module__ = "probnum.randprocs.covfuncs"
RatQuad.__module__ = "probnum.randprocs.covfuncs"
Matern.__module__ = "probnum.randprocs.covfuncs"
ProductMatern.__module__ = "probnum.randprocs.covfuncs"
