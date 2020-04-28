"""
Linear Algebra.

This package implements common operations and (probabilistic) numerical methods for linear algebra.
"""

from probnum.linalg.linearsolvers import *

# Public classes and functions. Order is reflected in documentation.
__all__ = ["problinsolve", "bayescg", "ProbabilisticLinearSolver", "GeneralMatrixBasedSolver",
           "SymmetricMatrixBasedSolver", "SolutionBasedSolver"]

# Set correct module paths (for superclasses). Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg"
GeneralMatrixBasedSolver.__module__ = "probnum.linalg"
