"""Stopping criteria of probabilistic linear solvers."""

from ._max_iterations import MaxIterations
from ._posterior_contraction import PosteriorContraction
from ._residual import Residual
from ._stopping_criterion import StoppingCriterion

# Public classes and functions. Order is reflected in documentation.
__all__ = ["StoppingCriterion", "MaxIterations", "Residual", "PosteriorContraction"]

# Set correct module paths. Corrects links and module paths in documentation.
StoppingCriterion.__module__ = "probnum.linalg.solvers.stop_criteria"
MaxIterations.__module__ = "probnum.linalg.solvers.stop_criteria"
Residual.__module__ = "probnum.linalg.solvers.stop_criteria"
PosteriorContraction.__module__ = "probnum.linalg.solvers.stop_criteria"
