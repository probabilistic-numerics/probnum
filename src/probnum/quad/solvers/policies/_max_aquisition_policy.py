"""Max acquisition policy for Bayesian quadrature."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from probnum.quad.solvers._bq_state import BQState
from probnum.quad.solvers.acquisition_functions import AcquisitionFunction
from probnum.quad.solvers.belief_updates import BQBeliefUpdate
from probnum.typing import IntLike

from ._policy import Policy

# pylint: disable=too-few-public-methods, fixme


class RandomMaxAcquisitionPolicy(Policy):
    """Policy that maximizer an acquisition function.

    Parameters
    ----------
    batch_size
        Size of batch of nodes when calling the policy once.
    acquisition_func
        The sample function. Needs to have the following interface:
        `sample_func(batch_size: int, rng: np.random.Generator)` and return an array of
        shape (batch_size, input_dim).
    n_candidates
        The number of candidate samples used to determine the maximizer.
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
        self,
        bq_state: BQState,
        belief_update: BQBeliefUpdate,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:

        random_nodes = bq_state.measure.sample(n_sample=self.n_candidates)
        values = self.acquisition_func(random_nodes, bq_state, belief_update)[0]
        idx_max = int(np.argmax(values))
        return random_nodes[idx_max, :][None, :]


class GreedyMaxAcquisitionPolicy(Policy):
    """Policy that maximizer an acquisition function.

    Parameters
    ----------
    batch_size
        Size of batch of nodes when calling the policy once.
    acquisition_func
        The sample function. Needs to have the following interface:
        `sample_func(batch_size: int, rng: np.random.Generator)` and return an array of
        shape (batch_size, input_dim).
    num_restarts
        The number of times the optimizer is being restarted.
    """

    def __init__(
        self,
        batch_size: IntLike,
        acquisition_func: AcquisitionFunction,
        num_restarts: IntLike,
    ) -> None:

        if batch_size != 1:
            raise ValueError(
                f"GreedyMaxAcquisitionPolicy can only be used with batch "
                f"size 1 ({batch_size})."
            )
        super().__init__(batch_size=batch_size)
        self.acquisition_func = acquisition_func
        self.num_restarts = int(num_restarts)

    @property
    def requires_rng(self) -> bool:
        return True

    def __call__(
        self,
        bq_state: BQState,
        belief_update: BQBeliefUpdate,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:

        measure = bq_state.measure
        domain = bq_state.measure.domain
        bounds = [(low, up) for (low, up) in zip(domain[0], domain[1])]

        f, jac = lambda x: self.acquisition_func(x[None, :], bq_state)
        f = lambda x: -f(x)[0]

        if self.acquisition_func.has_gradients:
            # Todo (#581): Gradients should not be available yet
            raise NotImplementedError
        else:
            initial_points = measure.sample(n_sample=self.num_restarts)
            for x0 in initial_points:
                scipy_minimize(
                    fun=f, jac=None, x0=x0, bounds=bounds, method="Nelder-Mead"
                )

        return bq_state
