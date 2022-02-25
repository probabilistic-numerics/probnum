"""Stopping criterion based on a maximum number of integrand evaluations."""

from probnum.quad.solvers.bq_state import BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion
from probnum.typing import IntLike

# pylint: disable=too-few-public-methods


class MaxNevals(BQStoppingCriterion):
    """Stop once a maximum number of integrand evaluations is reached.

    Parameters
    ----------
    max_nevals:
        Maximum number of integrand evaluations.
    """

    def __init__(self, max_nevals: IntLike):
        self.max_nevals = max_nevals

    def __call__(self, bq_state: BQState) -> bool:
        return bq_state.info.nevals >= self.max_nevals
