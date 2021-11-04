"""Probabilistic linear solvers.

Compositional implementation of probabilistic linear solvers. The classes and methods in
this subpackage allow the creation of custom iterative methods for the solution of
linear systems.
"""

from probnum.linalg.solvers.matrixbased import (
    MatrixBasedSolver,
    SymmetricMatrixBasedSolver,
)

from . import belief_updates, beliefs, information_ops, policies, stopping_criteria
from ._probabilistic_linear_solver import ProbabilisticLinearSolver
from ._state import LinearSolverState

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProbabilisticLinearSolver",
    "MatrixBasedSolver",
    "SymmetricMatrixBasedSolver",
    "LinearSolverState",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.solvers"
