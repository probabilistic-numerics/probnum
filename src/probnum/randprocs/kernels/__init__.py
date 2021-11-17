"""Kernels or covariance functions.

Kernels describe the spatial or temporal variation of a random process.
If evaluated at two sets of points a kernel is defined as the covariance
of the values of the random process at these locations.
"""

from ._exponentiated_quadratic import ExpQuad
from ._kernel import IsotropicMixin, Kernel
from ._linear import Linear
from ._matern import Matern
from ._polynomial import Polynomial
from ._rational_quadratic import RatQuad
from ._white_noise import WhiteNoise

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Kernel",
    "IsotropicMixin",
    "WhiteNoise",
    "Linear",
    "Polynomial",
    "ExpQuad",
    "RatQuad",
    "Matern",
]

# Set correct module paths. Corrects links and module paths in documentation.
Kernel.__module__ = "probnum.randprocs.kernels"
IsotropicMixin.__module__ = "probnum.randprocs.kernels"

WhiteNoise.__module__ = "probnum.randprocs.kernels"
Linear.__module__ = "probnum.randprocs.kernels"
Polynomial.__module__ = "probnum.randprocs.kernels"
ExpQuad.__module__ = "probnum.randprocs.kernels"
RatQuad.__module__ = "probnum.randprocs.kernels"
Matern.__module__ = "probnum.randprocs.kernels"
