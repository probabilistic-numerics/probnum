"""Integral variance reduction acquisition function for Bayesian quadrature."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from probnum.quad.solvers._bq_state import BQState
from probnum.quad.solvers.belief_updates import BQStandardBeliefUpdate

from ._acquisition_function import AcquisitionFunction

# pylint: disable=too-few-public-methods


class IntegralVarianceReduction(AcquisitionFunction):
    r"""The normalized reduction of the integral variance.

    The acquisition function is

    .. math::
        a(x) &= \mathfrak{v}^{-1}(\mathfrak{v} - \mathfrak{v}(x))\\
             &= \frac{(\int \bar{k}(x', x)p(x')\mathrm{d}x')^2}{\mathfrak{v} v(x)}\\
             &= \rho^2(x)

    where :math:`\mathfrak{v}` is the current integral variance, :math:`\mathfrak{v}(x)`
    is the integral variance including a hypothetical observation at
    :math:`x`, :math:`v(x)` is the predictive variance for :math:`f(x)` and
    :math:`\bar{k}(x', x)` is the posterior kernel function.

    The value :math:`a(x)` is equal to the squared correlation :math:`\rho^2(x)` between
    the hypothetical observation at :math:`x` and the integral value. [1]_

    The normalization constant :math:`\mathfrak{v}^{-1}` ensures that
    :math:`a(x)\in[0, 1]`.

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

        _, y_predictive_var = BQStandardBeliefUpdate.predict_integrand(x, bq_state)

        # if observation noise is added to BQ, it needs to be retrieved here.
        observation_noise_var = 0.0  # dummy placeholder
        y_predictive_var += observation_noise_var

        predictive_embedding = bq_state.kernel_embedding.kernel_mean(x)

        # posterior if observations are available
        if bq_state.fun_evals.shape[0] > 0:

            weights = BQStandardBeliefUpdate.gram_cho_solve(
                bq_state.gram_cho_factor, bq_state.kernel.matrix(bq_state.nodes, x)
            )
            predictive_embedding -= np.dot(bq_state.kernel_means, weights)

        values = (bq_state.scale_sq * predictive_embedding) ** 2 / (
            bq_state.integral_belief.cov * y_predictive_var
        )
        return values, None
