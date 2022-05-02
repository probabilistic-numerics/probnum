"""Stopping criterion based on the absolute value of the integral variance."""

from probnum.backend.typing import FloatLike
from probnum.quad.solvers.bq_state import BQIterInfo, BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion

# pylint: disable=too-few-public-methods


class IntegralVarianceTolerance(BQStoppingCriterion):
    """Stop once the integral variance is below some tolerance.

    Parameters
    ----------
    var_tol
        Tolerance value of the variance.
    """

    def __init__(self, var_tol: FloatLike):
        self.var_tol = var_tol

    def __call__(self, bq_state: BQState, info: BQIterInfo) -> bool:
        return bq_state.integral_belief.var <= self.var_tol
