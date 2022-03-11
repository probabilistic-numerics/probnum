"""Solution-based belief updates for the quantities of interest of a linear system."""

from ._projected_residual_belief_update import ProjectedResidualBeliefUpdate

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProjectedResidualBeliefUpdate",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProjectedResidualBeliefUpdate.__module__ = (
    "probnum.linalg.solvers.belief_updates.solution_based"
)
