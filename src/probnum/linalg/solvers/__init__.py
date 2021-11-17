"""Probabilistic linear solvers.

Compositional implementation of probabilistic linear solvers. The classes and methods in
this subpackage allow the creation of custom iterative methods for the solution of
linear systems.
"""

from . import belief_updates, beliefs, information_ops, policies, stopping_criteria
from ._probabilistic_linear_solver import (
    BayesCG,
    MatrixBasedPLS,
    ProbabilisticKaczmarz,
    ProbabilisticLinearSolver,
    SymMatrixBasedPLS,
)
from ._state import LinearSolverState

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProbabilisticLinearSolver",
    "LinearSolverState",
    "BayesCG",
    "ProbabilisticKaczmarz",
    "MatrixBasedPLS",
    "SymMatrixBasedPLS",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.solvers"
LinearSolverState.__module__ = "probnum.linalg.solvers"
BayesCG.__module__ = "probnum.linalg.solvers"
ProbabilisticKaczmarz.__module__ = "probnum.linalg.solvers"
MatrixBasedPLS.__module__ = "probnum.linalg.solvers"
SymMatrixBasedPLS.__module__ = "probnum.linalg.solvers"
