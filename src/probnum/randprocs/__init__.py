"""Random processes."""

from ._gauss_markov_process import GaussMarkovProcess
from ._gaussian_process import GaussianProcess
from ._random_process import RandomProcess

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "RandomProcess",
    "GaussianProcess",
    "GaussMarkovProcess",
]

# Set correct module paths. Corrects links and module paths in documentation.
RandomProcess.__module__ = "probnum.randprocs"

GaussianProcess.__module__ = "probnum.randprocs"
GaussMarkovProcess.__module__ = "probnum.randprocs"
