"""Belief updates for the quantities of interest of a linear system."""

from . import matrix_based, solution_based
from ._linear_solver_belief_update import LinearSolverBeliefUpdate

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverBeliefUpdate",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverBeliefUpdate.__module__ = "probnum.linalg.solvers.belief_updates"
