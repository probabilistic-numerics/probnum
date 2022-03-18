"""Stopping criteria for Bayesian quadrature methods."""

from ._bq_stopping_criterion import BQStoppingCriterion
from ._immediate_stop import ImmediateStop
from ._integral_variance_tol import IntegralVarianceTolerance
from ._max_nevals import MaxNevals
from ._rel_mean_change import RelativeMeanChange

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "BQStoppingCriterion",
    "ImmediateStop",
    "IntegralVarianceTolerance",
    "MaxNevals",
    "RelativeMeanChange",
]

# Set correct module paths. Corrects links and module paths in documentation.
BQStoppingCriterion.__module__ = "probnum.quad.solvers.stopping_criteria"
ImmediateStop.__module__ = "probnum.quad.solvers.stopping_criteria"
IntegralVarianceTolerance.__module__ = "probnum.quad.solvers.stopping_criteria"
MaxNevals.__module__ = "probnum.quad.solvers.stopping_criteria"
RelativeMeanChange.__module__ = "probnum.quad.solvers.stopping_criteria"
