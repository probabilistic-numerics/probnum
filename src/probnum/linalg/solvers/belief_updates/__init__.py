"""Belief updates for probabilistic linear solvers."""

from ._belief_update import LinearSolverBeliefUpdate
from ._symmetric_normal_linear_obs import SymmetricNormalLinearObsBeliefUpdate
from ._weak_mean_corr_linear_obs import WeakMeanCorrLinearObsBeliefUpdate

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverBeliefUpdate",
    "SymmetricNormalLinearObsBeliefUpdate",
    "WeakMeanCorrLinearObsBeliefUpdate",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverBeliefUpdate.__module__ = "probnum.linalg.solvers.belief_updates"
SymmetricNormalLinearObsBeliefUpdate.__module__ = (
    "probnum.linalg.solvers.belief_updates"
)
WeakMeanCorrLinearObsBeliefUpdate.__module__ = "probnum.linalg.solvers.belief_updates"
