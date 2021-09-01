"""Probabilistic Numerical Methods.

ProbNum implements probabilistic numerical methods in Python. Such
methods solve numerical problems from linear algebra, optimization,
quadrature and differential equations using probabilistic inference.
This approach captures uncertainty arising from finite computational
resources and stochastic input.
"""

# isort: off

# Global Configuration
# The global configuration registry. Can be used as a context manager to create local
# contexts in which configuration is temporarily overwritten. This object contains
# unguarded global state and is hence not thread-safe!
from ._config import _GLOBAL_CONFIG_SINGLETON as config

# isort: on

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
from ._version import version as __version__
from .randvars import asrandvar

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "asrandvar",
    "ProbabilisticNumericalMethod",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticNumericalMethod.__module__ = "probnum"
