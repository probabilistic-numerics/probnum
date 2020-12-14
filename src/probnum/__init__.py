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
| :mod:`~probnum.linalg`           | Probabilistic numerical linear algebra.                      |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.linops`           | Finite-dimensional linear operators.                         |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.quad`             | Bayesian quadrature / numerical integration.                 |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.random_variables` | Random variables representing uncertain values.              |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.problems`         | Definitions and collection of problems solved by PN methods. |
+----------------------------------+--------------------------------------------------------------+
| :mod:`~probnum.utils`            | Utility functions.                                           |
+----------------------------------+--------------------------------------------------------------+
"""

from pkg_resources import DistributionNotFound, get_distribution

from . import (
    diffeq,
    filtsmooth,
    linalg,
    linops,
    problems,
    quad,
    random_variables,
    utils,
)
from ._probabilistic_numerical_method import ProbabilisticNumericalMethod
from .random_variables import RandomVariable, asrandvar

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
