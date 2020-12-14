"""Probabilistic linear solvers.

Implementation and components of probabilistic linear solvers. The
classes and methods in this subpackage allow the creation of custom
linear solvers.
"""

from ._policies import (
    ConjugateDirectionsPolicy,
    ExploreExploitPolicy,
    LinearSolverPolicy,
)
from ._probabilistic_linear_solver import ProbabilisticLinearSolver
from ._stopping_criteria import (
    MaxIterations,
    PosteriorContraction,
    Residual,
    StoppingCriterion,
)

# from ._observe import
# from ._hyperparameter_optimization import
# from ._belief_updates import

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProbabilisticLinearSolver",
    "LinearSolverPolicy",
    "ConjugateDirectionsPolicy",
    "ExploreExploitPolicy",
    "StoppingCriterion",
    "MaxIterations",
    "Residual",
    "PosteriorContraction",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.linearsolvers"
LinearSolverPolicy.__module__ = "probnum.linalg.linearsolvers"
ConjugateDirectionsPolicy.__module__ = "probnum.linalg.linearsolvers"
ExploreExploitPolicy.__module__ = "probnum.linalg.linearsolvers"
StoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
MaxIterations.__module__ = "probnum.linalg.linearsolvers"
Residual.__module__ = "probnum.linalg.linearsolvers"
PosteriorContraction.__module__ = "probnum.linalg.linearsolvers"
