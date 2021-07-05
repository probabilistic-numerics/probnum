"""Belief classes for the quantities of interest of a linear system."""

from ._linear_system_belief import LinearSystemBelief

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSystemBelief"]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSystemBelief.__module__ = "probnum.linalg.solvers.beliefs"
