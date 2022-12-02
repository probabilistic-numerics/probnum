"""Uncertainty sampling for Bayesian Monte Carlo."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from probnum.quad.solvers._bq_state import BQState

from ._acquisition_function import AcquisitionFunction

# pylint: disable=too-few-public-methods, fixme


class WeightedPredictiveVariance(AcquisitionFunction):
    r"""The predictive variance acquisition function that yields uncertainty sampling.

    .. math::
        a(x) = \operatorname{Var}(f(x)) p(x)^2

    where :math:`\operatorname{Var}(f(x))` is the predictive variance and :math:`p(x)`
    is the density of the integration measure :math:`\mu`.

    """

    @property
    def has_gradients(self) -> bool:
        # Todo (#581): this needs to return True, once gradients are available
        return False

    def __call__(
        self, x: np.ndarray, bq_state: BQState
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        weights = bq_state.measure(x) ** 2
        k = bq_state.kernel

        # Todo: implement
        predicitve_variance = weights
        return weights * predicitve_variance, None
