"""Stopping criterion that stops immediately."""

from probnum.quad.solvers.bq_state import BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion

# pylint: disable=too-few-public-methods


class ImmediateStop(BQStoppingCriterion):
    """Dummy stopping criterion that always stops. This is useful for fixed datasets
    when no policy or acquisition loop is required or given.
    """

    def __call__(self, bq_state: BQState) -> bool:
        return True
