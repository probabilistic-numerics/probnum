"""Stopping criteria for Bayesian quadrature methods."""

from ._bq_stopping_criterion import BQStoppingCriterion
from ._integral_variance_tol import IntegralVarianceToleranceStoppingCriterion
from ._max_nevals import MaxNevalsStoppingCriterion
from ._rel_mean_change import RelativeMeanChangeStoppingCriterion

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "BQStoppingCriterion",
    "IntegralVarianceToleranceStoppingCriterion",
    "MaxNevalsStoppingCriterion",
    "RelativeMeanChangeStoppingCriterion",
]

# Set correct module paths. Corrects links and module paths in documentation.
BQStoppingCriterion.__module__ = "probnum.quad.solvers.stopping_criteria"
IntegralVarianceToleranceStoppingCriterion.__module__ = (
    "probnum.quad.solvers.stopping_criteria"
)
MaxNevalsStoppingCriterion.__module__ = "probnum.quad.solvers.stopping_criteria"
RelativeMeanChangeStoppingCriterion.__module__ = (
    "probnum.quad.solvers.stopping_criteria"
)
