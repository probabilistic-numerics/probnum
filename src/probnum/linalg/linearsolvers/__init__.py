"""Probabilistic linear solvers.

Implementation and components of probabilistic linear solvers. The
classes and methods in this subpackage allow the creation of custom
linear solvers.
"""

from ._linear_solver_state import LinearSolverState
from ._policies import (
    ConjugateDirectionsPolicy,
    ExploreExploitPolicy,
    LinearSolverPolicy,
)
from ._probabilistic_linear_solver import ProbabilisticLinearSolver
from ._stopping_criteria import (
    MaxIterStoppingCriterion,
    PosteriorStoppingCriterion,
    ResidualStoppingCriterion,
    StoppingCriterion,
)

# from ._observe import
# from ._hyperparameter_optimization import
# from ._belief_updates import

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProbabilisticLinearSolver",
    "LinearSolverState",
    "LinearSolverPolicy",
    "ConjugateDirectionsPolicy",
    "ExploreExploitPolicy",
    "StoppingCriterion",
    "MaxIterStoppingCriterion",
    "ResidualStoppingCriterion",
    "PosteriorStoppingCriterion",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.linearsolvers"
LinearSolverState.__module__ = "probnum.linalg.linearsolvers"
LinearSolverPolicy.__module__ = "probnum.linalg.linearsolvers"
ConjugateDirectionsPolicy.__module__ = "probnum.linalg.linearsolvers"
ExploreExploitPolicy.__module__ = "probnum.linalg.linearsolvers"
StoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
MaxIterStoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
ResidualStoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
PosteriorStoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
