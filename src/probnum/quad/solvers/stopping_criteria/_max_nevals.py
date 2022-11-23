"""Stopping criterion based on a maximum number of integrand evaluations."""

from __future__ import annotations

from probnum.quad.solvers._bq_state import BQIterInfo, BQState
from probnum.quad.solvers.stopping_criteria import BQStoppingCriterion
from probnum.typing import IntLike

# pylint: disable=too-few-public-methods


class MaxNevals(BQStoppingCriterion):
    """Stop once a maximum number of integrand evaluations is reached.

    Parameters
    ----------
    max_nevals
        Maximum number of integrand evaluations.
    """

    def __init__(self, max_nevals: IntLike):
        self.max_nevals = int(max_nevals)

    def __call__(self, bq_state: BQState, info: BQIterInfo) -> bool:
        return info.nevals >= self.max_nevals
