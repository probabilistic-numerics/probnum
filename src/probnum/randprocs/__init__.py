"""Random processes."""

from ._gaussian_process import GaussianProcess
from ._markov_process import MarkovProcess
from ._random_process import RandomProcess

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "RandomProcess",
    "MarkovProcess",
    "GaussianProcess",
]

# Set correct module paths. Corrects links and module paths in documentation.
RandomProcess.__module__ = "probnum.randprocs"
MarkovProcess.__module__ = "probnum.randprocs"
GaussianProcess.__module__ = "probnum.randprocs"
