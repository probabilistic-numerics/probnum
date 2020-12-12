"""Linear Algebra."""
from probnum.linalg.linearsolvers import (
    ProbabilisticLinearSolver,
    bayescg,
    problinsolve,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "problinsolve",
    "bayescg",
    "ProbabilisticLinearSolver",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg"
