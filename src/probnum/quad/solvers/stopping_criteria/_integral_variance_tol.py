"""Stopping criterion based on the absolute value of the integral variance"""

from probnum.quad.solvers.bq_state import BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion
from probnum.randvars import Normal
from probnum.typing import FloatArgType


class IntegralVarianceTolerance(BQStoppingCriterion):
    """Stop once the integral variance is below some tolerance.

    Parameters
    ----------
    var_tol:
        Tolerance value of the variance.
    """

    def __init__(self, var_tol: FloatArgType):
        self.var_tol = var_tol

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        return integral_belief.var <= self.var_tol
