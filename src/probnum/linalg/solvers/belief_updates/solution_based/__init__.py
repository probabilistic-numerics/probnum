"""Solution-based belief updates for the quantities of interest of a linear system."""

from ._solution_based_proj_rhs_belief_update import (
    SolutionBasedProjectedRHSBeliefUpdate,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "SolutionBasedProjectedRHSBeliefUpdate",
]

# Set correct module paths. Corrects links and module paths in documentation.
SolutionBasedProjectedRHSBeliefUpdate.__module__ = (
    "probnum.linalg.solvers.belief_updates.solution_based"
)
