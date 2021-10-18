"""Belief updates for the quantities of interest of a linear system."""

from ._linear_system_belief_update import LinearSystemBeliefUpdate

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSystemBeliefUpdate"]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSystemBeliefUpdate.__module__ = "probnum.linalg.solvers.belief_updates"
