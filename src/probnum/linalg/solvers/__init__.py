"""Probabilistic linear solvers.

Compositional implementation of probabilistic linear solvers. The
classes and methods in this subpackage allow the creation of custom
iterative methods for the solution of linear systems.
"""

from probnum.linalg.solvers.matrixbased import (
    MatrixBasedSolver,
    ProbabilisticLinearSolver,
    SymmetricMatrixBasedSolver,
)

from . import beliefs, policies
from ._state import LinearSolverState

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProbabilisticLinearSolver",
    "LinearSolverState",
    "MatrixBasedSolver",
    "SymmetricMatrixBasedSolver",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.solvers"
LinearSolverState.__module__ = "probnum.linalg.solvers"
