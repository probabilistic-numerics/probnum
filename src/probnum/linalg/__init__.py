"""Linear Algebra."""
from probnum.linalg.linearsolvers import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "problinsolve",
    "bayescg",
    "ProbabilisticLinearSolver",
    "MatrixBasedSolver",
    "AsymmetricMatrixBasedSolver",
    "SymmetricMatrixBasedSolver",
    "SolutionBasedSolver",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg"
MatrixBasedSolver.__module__ = "probnum.linalg"
