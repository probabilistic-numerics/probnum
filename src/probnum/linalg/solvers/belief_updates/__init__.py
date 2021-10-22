"""Belief updates for the quantities of interest of a linear system."""

from ._linear_solver_belief_update import LinearSolverBeliefUpdate
from ._matrix_based_linear_belief_update import MatrixBasedLinearBeliefUpdate
from ._solution_based_proj_rhs_belief_update import (
    SolutionBasedProjectedRHSBeliefUpdate,
)
from ._symmetric_matrix_based_linear_belief_update import (
    SymmetricMatrixBasedLinearBeliefUpdate,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverBeliefUpdate",
    "SolutionBasedProjectedRHSBeliefUpdate",
    "MatrixBasedLinearBeliefUpdate",
    "SymmetricMatrixBasedLinearBeliefUpdate",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverBeliefUpdate.__module__ = "probnum.linalg.solvers.belief_updates"
SolutionBasedProjectedRHSBeliefUpdate.__module__ = (
    "probnum.linalg.solvers.belief_updates"
)
MatrixBasedLinearBeliefUpdate.__module__ = "probnum.linalg.solvers.belief_updates"

SymmetricMatrixBasedLinearBeliefUpdate.__module__ = (
    "probnum.linalg.solvers.belief_updates"
)
