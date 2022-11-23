"""Belief updates for Bayesian quadrature."""

from ._belief_update import BQBeliefUpdate
from ._standard_update import BQStandardBeliefUpdate

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "BQBeliefUpdate",
    "BQStandardBeliefUpdate",
]

# Set correct module paths. Corrects links and module paths in documentation.
BQBeliefUpdate.__module__ = "probnum.quad.solvers.belief_updates"
BQStandardBeliefUpdate.__module__ = "probnum.quad.solvers.belief_updates"
