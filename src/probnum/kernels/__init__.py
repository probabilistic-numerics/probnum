"""Kernels or covariance functions."""

from ._exponentiated_quadratic import ExpQuad
from ._kernel import Kernel
from ._linear import Linear
from ._matern import Matern
from ._polynomial import Polynomial
from ._rational_quadratic import RatQuad
from ._white_noise import WhiteNoise

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Kernel",
    "WhiteNoise",
    "Linear",
    "Polynomial",
    "ExpQuad",
    "RatQuad",
    "Matern",
]

# Set correct module paths. Corrects links and module paths in documentation.
Kernel.__module__ = "probnum.kernels"
WhiteNoise.__module__ = "probnum.kernels"
Linear.__module__ = "probnum.kernels"
Polynomial.__module__ = "probnum.kernels"
ExpQuad.__module__ = "probnum.kernels"
RatQuad.__module__ = "probnum.kernels"
Matern.__module__ = "probnum.kernels"
