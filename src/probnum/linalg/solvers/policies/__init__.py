"""Policies of probabilistic linear solvers returning actions."""

from ._policy import Policy

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Policy",
]

# Set correct module paths. Corrects links and module paths in documentation.
Policy.__module__ = "probnum.linalg.solvers.policies"
