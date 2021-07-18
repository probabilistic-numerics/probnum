"""Probabilistic Numerical Methods.

ProbNum implements probabilistic numerical methods in Python. Such
methods solve numerical problems from linear algebra, optimization,
quadrature and differential equations using probabilistic inference.
This approach captures uncertainty arising from finite computational
resources and stochastic input.
"""

from pkg_resources import DistributionNotFound, get_distribution

# Global Configuration
from ._config import _GLOBAL_CONFIG_SINGLETON as config

"""The global configuration registry. Can be used as a context manager to create local
contexts in which configuration is temporarily overwritten. This object contains
unguarded global state and is hence not thread-safe!"""

from . import (
    diffeq,
    filtsmooth,
    kernels,
    linalg,
    linops,
    problems,
    quad,
    randprocs,
    randvars,
    utils,
)
from ._probabilistic_numerical_method import ProbabilisticNumericalMethod
from .randvars import asrandvar

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "asrandvar",
    "ProbabilisticNumericalMethod",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticNumericalMethod.__module__ = "probnum"

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
