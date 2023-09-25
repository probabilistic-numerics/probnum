"""Test problems for numerical integration/ quadrature."""

from ._emukit_problems import circulargaussian2d, hennig1d, hennig2d, sombrero2d

from ._quadproblems_gaussian import *
from ._quadproblems_uniform import *

# Public classes and functions. Order is reflected in documentation.
__all__ = ["circulargaussian2d", "hennig1d", "hennig2d", "sombrero2d"]
