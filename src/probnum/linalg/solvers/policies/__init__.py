"""Policies of probabilistic linear solvers returning actions."""

from ._conjugate_gradient import ConjugateGradientPolicy
from ._linear_solver_policy import LinearSolverPolicy
from ._random_unit_vector import RandomUnitVectorPolicy

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSolverPolicy", "ConjugateGradientPolicy", "RandomUnitVectorPolicy"]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverPolicy.__module__ = "probnum.linalg.solvers.policies"
RandomUnitVectorPolicy.__module__ = "probnum.linalg.solvers.policies"
ConjugateGradientPolicy.__module__ = "probnum.linalg.solvers.policies"
