"""Probabilistic linear solvers.

Implementation and components of probabilistic linear solvers. The
classes and methods in this subpackage allow the creation of custom
linear solvers.
"""

from ._belief_updates import BeliefUpdate, LinearGaussianBeliefUpdate
from ._observation_operators import MatrixMultObservation, ObservationOperator
from ._policies import ConjugateDirectionsPolicy, ExploreExploitPolicy, Policy
from ._probabilistic_linear_solver import LinearSolverState, ProbabilisticLinearSolver
from ._stopping_criteria import (
    MaxIterStoppingCriterion,
    PosteriorStoppingCriterion,
    ResidualStoppingCriterion,
    StoppingCriterion,
)

# from ._hyperparameter_optimization import

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ProbabilisticLinearSolver",
    "LinearSolverState",
    "Policy",
    "ConjugateDirectionsPolicy",
    "ExploreExploitPolicy",
    "StoppingCriterion",
    "MaxIterStoppingCriterion",
    "ResidualStoppingCriterion",
    "PosteriorStoppingCriterion",
    "ObservationOperator",
    "MatrixMultObservation",
    "BeliefUpdate",
    "LinearGaussianBeliefUpdate",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticLinearSolver.__module__ = "probnum.linalg.linearsolvers"
LinearSolverState.__module__ = "probnum.linalg.linearsolvers"
Policy.__module__ = "probnum.linalg.linearsolvers"
ConjugateDirectionsPolicy.__module__ = "probnum.linalg.linearsolvers"
ExploreExploitPolicy.__module__ = "probnum.linalg.linearsolvers"
StoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
MaxIterStoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
ResidualStoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
PosteriorStoppingCriterion.__module__ = "probnum.linalg.linearsolvers"
ObservationOperator.__module__ = "probnum.linalg.linearsolvers"
MatrixMultObservation.__module__ = "probnum.linalg.linearsolvers"
BeliefUpdate.__module__ = "probnum.linalg.linearsolvers"
LinearGaussianBeliefUpdate.__module__ = "probnum.linalg.linearsolvers"
