"""Probabilistic linear solvers.

Implementation and components of probabilistic linear solvers. The
classes and methods in this subpackage allow the creation of custom
linear solvers.
"""

from ._probabilistic_linear_solver import LinearSolverState, ProbabilisticLinearSolver

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ProbabilisticLinearSolver", "LinearSolverState"]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.linearsolvers"
LinearSolverState.__module__ = "probnum.linalg.linearsolvers"
