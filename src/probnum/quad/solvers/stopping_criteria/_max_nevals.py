"""Stopping criterion based on a maximum number of integrand evaluations."""

from probnum.backend.typing import IntLike
from probnum.quad.solvers.bq_state import BQIterInfo, BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion

# pylint: disable=too-few-public-methods


class MaxNevals(BQStoppingCriterion):
    """Stop once a maximum number of integrand evaluations is reached.

    Parameters
    ----------
    max_nevals
        Maximum number of integrand evaluations.
    """

    def __init__(self, max_nevals: IntLike):
        self.max_nevals = max_nevals

    def __call__(self, bq_state: BQState, info: BQIterInfo) -> bool:
        return info.nevals >= self.max_nevals
