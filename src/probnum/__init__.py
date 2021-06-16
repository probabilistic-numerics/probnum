"""ProbNum implements probabilistic numerical methods in Python. Such methods solve
numerical problems from linear algebra, optimization, quadrature and differential
equations using probabilistic inference. This approach captures uncertainty arising from
finite computational resources and stochastic input.

+----------------------------------+--------------------------------------------------------------+
| **Subpackage**                   | **Description**                                              |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.diffeq`           | Probabilistic solvers for ordinary differential equations.   |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.filtsmooth`       | Bayesian filtering and smoothing.                            |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.kernels`          | Kernels / covariance functions.                              |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.linalg`           | Probabilistic numerical linear algebra.                      |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.linops`           | Finite-dimensional linear operators.                         |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.problems`         | Definitions and collection of problems solved by PN methods. |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.quad`             | Bayesian quadrature / numerical integration.                 |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.randprocs`        | Random processes representing uncertain functions.           |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.randvars`         | Random variables representing uncertain values.              |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.statespace`       | Probabilistic state space models.                            |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.utils`            | Utility functions.                                           |
+----------------------------------+--------------------------------------------------------------+
"""

from pkg_resources import DistributionNotFound, get_distribution

from . import (  # diffeq,; filtsmooth,
    kernels,
    linalg,
    linops,
    problems,
    quad,
    randprocs,
    randvars,
    statespace,
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
