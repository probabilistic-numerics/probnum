"""Stopping criterion based on a maximum number of iteration."""

from probnum.quad.solvers.bq_state import BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion
from probnum.randvars import Normal
from probnum.typing import IntArgType

# pylint: disable=too-few-public-methods


class MaxNevals(BQStoppingCriterion):
    """Stop once a maximum number of iterations is reached.

    Parameters
    ----------
    max_evals:
        Maximum number of function evaluations.
    """

    def __init__(self, max_evals: IntArgType):
        self.max_evals = max_evals

    def __call__(self, integral_belief: Normal, bq_state: BQState) -> bool:
        return bq_state.info.nevals >= self.max_evals
