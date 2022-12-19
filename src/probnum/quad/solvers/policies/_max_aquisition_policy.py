"""Max acquisition policy for Bayesian quadrature."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize as scipy_minimize

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
                f"The number of candidates ({n_candidates}) must be equal to"
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


class MaxAcquisitionPolicy(Policy):
    """Policy that maximizes an acquisition function with an optimizer.

    Parameters
    ----------
    batch_size
        Size of batch of nodes when calling the policy once (must be equal to 1).
    acquisition_func
        The acquisition function.
    n_restarts
        The number of times the optimizer is being restarted.

    Raises
    ------
    ValueError
        If ``batch_size`` is not 1, or if ``n_restarts`` is too small.

    Notes
    -----
    The policy uses SciPy's 'Nelder-Mead' optimizer when gradients are unavailable.
    This is the current standard setting since ``probnum`` does not provide gradients
    yet.

    """

    def __init__(
        self,
        batch_size: IntLike,
        acquisition_func: AcquisitionFunction,
        n_restarts: IntLike,
    ) -> None:

        if batch_size != 1:
            raise ValueError(
                f"MaxAcquisitionPolicy can only be used with batch size 1 "
                f"({batch_size})."
            )
        if n_restarts < 1:
            raise ValueError(
                f"The number of restarts ({n_restarts}) must be equal to"
                f"or larger than 1."
            )

        super().__init__(batch_size=batch_size)
        self.acquisition_func = acquisition_func
        self.n_restarts = int(n_restarts)

    @property
    def requires_rng(self) -> bool:
        return True

    def __call__(
        self, bq_state: BQState, rng: Optional[np.random.Generator]
    ) -> np.ndarray:

        measure = bq_state.measure
        domain = bq_state.measure.domain
        bounds = list(zip(domain[0], domain[1]))

        if self.acquisition_func.has_gradients:
            # Todo (#581): Gradients should not be available yet.
            raise NotImplementedError

        f = lambda x: -self.acquisition_func(x[None, :], bq_state)[0][0]
        initial_points = measure.sample(n_sample=self.n_restarts, rng=rng)
        x_min, f_min, success = None, np.inf, False
        for x0 in initial_points:
            scipy_result = scipy_minimize(
                fun=f, jac=None, x0=x0, bounds=bounds, method="Nelder-Mead"
            )
            if scipy_result["success"] and scipy_result["fun"] < f_min:
                f_min = scipy_result["fun"]
                x_min = scipy_result["x"][None, :]
                success = True

        if not success:
            raise RuntimeError(
                f"Acquisition optimizer could not find a suitable maximizer with "
                f"({self.n_restarts}) restarts."
            )

        return x_min
