"""Stopping criteria for probabilistic linear solvers."""

from ._linear_solver_stopping_criterion import LinearSolverStopCrit
from ._maxiter import MaxIterationsStopCrit
from ._posterior_contraction import PosteriorContractionStopCrit
from ._residual_norm import ResidualNormStopCrit

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "LinearSolverStopCrit",
    "MaxIterationsStopCrit",
    "ResidualNormStopCrit",
    "PosteriorContractionStopCrit",
]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverStopCrit.__module__ = "probnum.linalg.solvers.stopping_criteria"
MaxIterationsStopCrit.__module__ = "probnum.linalg.solvers.stopping_criteria"
ResidualNormStopCrit.__module__ = "probnum.linalg.solvers.stopping_criteria"
PosteriorContractionStopCrit.__module__ = "probnum.linalg.solvers.stopping_criteria"
