"""Stopping criterion that stops immediately."""

from probnum.quad.solvers._bq_state import BQIterInfo, BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion

# pylint: disable=too-few-public-methods


class ImmediateStop(BQStoppingCriterion):
    """Dummy stopping criterion that always stops. This is useful for fixed datasets
    when no policy or acquisition loop is required or given.

    """

    def __call__(self, bq_state: BQState, info: BQIterInfo) -> bool:
        return True
