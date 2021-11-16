"""Stopping criterion based on the relative change of the successive integral estimators."""

import numpy as np

from probnum.quad.solvers.bq_state import BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion
from probnum.randvars import Normal
from probnum.typing import FloatArgType

# pylint: disable=too-few-public-methods


class RelativeMeanChange(BQStoppingCriterion):
    """Stop once the relative change of consecutive integral estimates are smaller than
    a tolerance. That is, the stopping criterion is.

        | current_integral_estimate - previous_integral_estimate) /
      current_integral_estimate | <= rel_tol.

    Parameters
    ----------
    rel_tol:
        Relative error tolerance on consecutive integral mean values.
    """

    def __init__(self, rel_tol: FloatArgType):
        self.rel_tol = rel_tol

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        return (
            np.abs(
                (integral_belief.mean - bq_state.previous_integral_beliefs[-1].mean)
                / integral_belief.mean
            )
            <= self.rel_tol
        )
