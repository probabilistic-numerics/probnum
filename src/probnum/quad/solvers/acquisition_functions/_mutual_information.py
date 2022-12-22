"""Mutual information acquisition function for Bayesian quadrature."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from probnum.quad.solvers._bq_state import BQState

from ._acquisition_function import AcquisitionFunction
from ._integral_variance_reduction import IntegralVarianceReduction

# pylint: disable=too-few-public-methods


class MutualInformation(AcquisitionFunction):
    r"""The mutual information between a hypothetical integrand observation and the
    integral value.

    The acquisition function is

    .. math::
        a(x) = -0.5 \log(1-\rho^2(x))

    where :math:`\rho^2(x)` is the squared correlation between a hypothetical integrand
    observations at :math:`x` and the integral value. [1]_

    The mutual information is non-negative and unbounded for a 'perfect' observation
    and :math:`\rho^2(x) = 1.`

    References
    ----------
    .. [1] Gessner et al. Active Multi-Information Source Bayesian Quadrature,
       *UAI*, 2019

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
        ivr = IntegralVarianceReduction()
        rho2, _ = ivr(x, bq_state)
        values = -0.5 * np.log(1 - rho2)
        return values, None
