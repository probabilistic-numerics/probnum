"""Example Implementation of 1D (noisy) quadratic optimization."""

from ._probsolve_qp import probsolve_qp
from .probabilistic_quadratic_optimizer import ProbabilisticQuadraticOptimizer

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "probsolve_qp",
    "ProbabilisticQuadraticOptimizer",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticQuadraticOptimizer.__module__ = "probnum.quadopt_example"
