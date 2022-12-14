"""Uncertainty sampling for Bayesian Monte Carlo."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from probnum.quad.solvers._bq_state import BQState
from probnum.quad.solvers.belief_updates import BQStandardBeliefUpdate

from ._acquisition_function import AcquisitionFunction

# pylint: disable=too-few-public-methods, fixme


class WeightedPredictiveVariance(AcquisitionFunction):
    r"""The predictive variance acquisition function that yields uncertainty sampling.

    The acquisition function is

    .. math::
        a(x) = \operatorname{Var}(f(x)) p(x)^2

    where :math:`\operatorname{Var}(f(x))` is the predictive variance of the model and
    :math:`p(x)` is the density of the integration measure :math:`\mu`.

    """

    @property
    def has_gradients(self) -> bool:
        # Todo (#581): this needs to return True, once gradients are available
        return False

    def __call__(
        self,
        x: np.ndarray,
        bq_state: BQState,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        predictive_variance = bq_state.kernel(x, x)
        if bq_state.fun_evals.shape != (0,):
            kXx = bq_state.kernel.matrix(bq_state.nodes, x)
            regression_weights = BQStandardBeliefUpdate.gram_cho_solve(
                bq_state.gram_cho_factor, kXx
            )
            predictive_variance -= np.sum(regression_weights * kXx, axis=0)
        values = bq_state.scale_sq * predictive_variance * bq_state.measure(x) ** 2
        return values, None
