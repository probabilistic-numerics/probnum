"""Random policy for Bayesian quadrature."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from probnum.quad.solvers._bq_state import BQState
from probnum.typing import IntLike

from ._policy import Policy

# pylint: disable=too-few-public-methods, fixme


class RandomPolicy(Policy):
    """Random sampling from an objective.

    Parameters
    ----------
    batch_size
        Size of batch of nodes when calling the policy once.
    sample_func
        The sample function. Needs to have the following interface:
        `sample_func(batch_size: int, rng: np.random.Generator)` and return an array of
        shape (batch_size, input_dim).
    """

    def __init__(
        self,
        batch_size: IntLike,
        sample_func: Callable,
    ) -> None:
        super().__init__(batch_size=batch_size)
        self.sample_func = sample_func

    @property
    def requires_rng(self) -> bool:
        return True

    def __call__(
        self, bq_state: BQState, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        return self.sample_func(self.batch_size, rng=rng)
