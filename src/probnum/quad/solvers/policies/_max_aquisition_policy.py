"""Max acquisition policy for Bayesian quadrature."""

from __future__ import annotations

from typing import Optional

import numpy as np

from probnum.quad.solvers._bq_state import BQState
from probnum.quad.solvers.acquisition_functions import AcquisitionFunction
from probnum.typing import IntLike

from ._policy import Policy

# pylint: disable=too-few-public-methods, fixme


class RandomMaxAcquisitionPolicy(Policy):
    """Policy that maximizes an acquisition function by sampling random candidate nodes.

    The candidate nodes are random draws from the integration measure. The node with the
    largest acquisition value is chosen.

    Parameters
    ----------
    batch_size
        Size of batch of nodes when calling the policy once (must be equal to 1).
    acquisition_func
        The acquisition function.
    n_candidates
        The number of candidate nodes.

    Raises
    ------
    ValueError
        If ``batch_size`` is not 1, or if ``n_candidates`` is too small.
    """

    def __init__(
        self,
        batch_size: IntLike,
        acquisition_func: AcquisitionFunction,
        n_candidates: IntLike,
    ) -> None:

        if batch_size != 1:
            raise ValueError(
                f"RandomMaxAcquisitionPolicy can only be used with batch "
                f"size 1 ({batch_size})."
            )
        if n_candidates < 1:
            raise ValueError(
                f"The number of candidates ({n_candidates}) must be equal "
                f"or larger than 1."
            )

        super().__init__(batch_size=batch_size)
        self.acquisition_func = acquisition_func
        self.n_candidates = int(n_candidates)

    @property
    def requires_rng(self) -> bool:
        return True

    def __call__(
        self, bq_state: BQState, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        random_nodes = bq_state.measure.sample(n_sample=self.n_candidates, rng=rng)
        values = self.acquisition_func(random_nodes, bq_state)[0]
        idx_max = int(np.argmax(values))
        return random_nodes[idx_max, :][None, :]
