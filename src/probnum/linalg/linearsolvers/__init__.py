"""Probabilistic linear solvers.

Implementation and components of probabilistic linear solvers. The
classes and methods in this subpackage allow the creation of custom
linear solvers.
"""

from ._policies import conjugate_directions_policy, explore_exploit_policy
from ._probabilistic_linear_solver import ProbabilisticLinearSolver

# from ._stopping_criteria import
# from ._observe import
# from ._hyperparameter_optimization import
# from ._belief_updates import

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProbabilisticLinearSolver",
    "conjugate_directions_policy",
    "explore_exploit_policy",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.linearsolvers"
