"""Random Processes.

Random processes generalize functions by encoding uncertainty over
function values in their covariance function. They can be used to model
(deterministic) functions which are not fully known or to define
functions with stochastic output.
"""

from ._gaussian_process import GaussianProcess
from ._random_process import RandomProcess

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "RandomProcess",
    "GaussianProcess",
]
from . import markov

# Set correct module paths. Corrects links and module paths in documentation.
RandomProcess.__module__ = "probnum.randprocs"
GaussianProcess.__module__ = "probnum.randprocs"
