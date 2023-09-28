"""Test problems for numerical integration/ quadrature."""

from ._emukit_problems import circulargaussian2d, hennig1d, hennig2d, sombrero2d
from ._quadproblems_gaussian import *
from ._quadproblems_uniform import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "circulargaussian2d",
    "hennig1d",
    "hennig2d",
    "sombrero2d",
    "uniform_to_gaussian_quadprob",
    "sum_polynomials",
    "genz_continuous",
    "genz_cornerpeak",
    "genz_discontinuous",
    "genz_gaussian",
    "genz_oscillatory",
    "genz_productpeak",
    "bratley1992",
    "roos_arnold",
    "gfunction",
    "morokoff_caflisch_1",
    "morokoff_caflisch_2",
]
