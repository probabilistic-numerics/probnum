"""Linear Algebra.

This package implements probabilistic numerical methods for the solution
of problems arising in linear algebra, such as the solution of linear
systems.
"""
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
