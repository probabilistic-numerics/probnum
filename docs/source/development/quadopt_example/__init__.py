"""
Example Implementation of 1D (noisy) quadratic optimization.
"""

from ._quadratic_programming import ProbabilisticQuadraticOptimizer, probsolve_qp

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "probsolve_qp",
    "ProbabilisticQuadraticOptimizer",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticQuadraticOptimizer.__module__ = "probnum.quadopt_example"
