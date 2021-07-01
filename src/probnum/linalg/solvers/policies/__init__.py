"""Policies of probabilistic linear solvers returning actions."""

from ._conjugate_gradient import ConjugateGradient
from ._policy import Policy
from ._standard_unit_vectors import StandardUnitVectors

# Public classes and functions. Order is reflected in documentation.
__all__ = ["Policy", "StandardUnitVectors", "ConjugateGradient"]

# Set correct module paths. Corrects links and module paths in documentation.
Policy.__module__ = "probnum.linalg.solvers.policies"
StandardUnitVectors.__module__ = "probnum.linalg.solvers.policies"
ConjugateGradient.__module__ = "probnum.linalg.solvers.policies"
