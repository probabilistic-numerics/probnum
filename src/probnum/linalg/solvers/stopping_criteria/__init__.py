"""Stopping criteria for probabilistic linear solvers."""

from ._linear_solver_stopping_criterion import LinearSolverStoppingCriterion
from ._maxiter import MaxIterationsStoppingCriterion
from ._posterior_contraction import PosteriorContractionStoppingCriterion
from ._residual_norm import ResidualNormStoppingCriterion

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverStoppingCriterion",
    "MaxIterationsStoppingCriterion",
    "ResidualNormStoppingCriterion",
    "PosteriorContractionStoppingCriterion",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverStoppingCriterion.__module__ = "probnum.linalg.solvers.stopping_criteria"
MaxIterationsStoppingCriterion.__module__ = "probnum.linalg.solvers.stopping_criteria"
ResidualNormStoppingCriterion.__module__ = "probnum.linalg.solvers.stopping_criteria"
PosteriorContractionStoppingCriterion.__module__ = (
    "probnum.linalg.solvers.stopping_criteria"
)
