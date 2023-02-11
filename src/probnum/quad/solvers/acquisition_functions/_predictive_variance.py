"""Uncertainty sampling for Bayesian quadrature."""

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

    Notes
    -----
        The implementation scales :math:`a(x)` with the inverse of the squared kernel
        scale for numerical stability.
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

        _, predictive_variance = BQStandardBeliefUpdate.predict_integrand(x, bq_state)
        predictive_variance *= 1 / bq_state.scale_sq  # for numerical stability

        values = predictive_variance * bq_state.measure(x) ** 2
        return values, None
