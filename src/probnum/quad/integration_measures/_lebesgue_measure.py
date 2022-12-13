"""The Lebesgue measure."""


from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.stats

from probnum.quad.typing import DomainLike
from probnum.typing import IntLike

from ._integration_measure import IntegrationMeasure


class LebesgueMeasure(IntegrationMeasure):
    """Lebesgue measure on a hyper-rectangle.

    Parameters
    ----------
    domain
        Domain of integration. Contains lower and upper bound as a scalar or
        ``np.ndarray``.
    input_dim
        Dimension of the integration domain. If not given, inferred from ``domain``.
    normalized
         Boolean which controls whether the measure is normalized (i.e.,
         integral over the domain is one). Defaults to ``False``.
    """

    def __init__(
        self,
        domain: DomainLike,
        input_dim: Optional[IntLike] = None,
        normalized: bool = False,
    ) -> None:
        super().__init__(input_dim=input_dim, domain=domain)

        # Set normalization constant
        if normalized:
            normalization_constant = 1.0 / np.prod(self.domain[1] - self.domain[0])
        else:
            normalization_constant = 1.0

        if normalization_constant in [0, np.Inf, -np.Inf]:
            raise ValueError(
                "Normalization constant is too small or too large. "
                "Consider setting normalized = False."
            )

        self.normalized = normalized
        self.normalization_constant = normalization_constant

        # Use scipy's uniform random variable since uniform random variables are not
        # yet implemented in probnum
        self.random_variable = scipy.stats.uniform(
            loc=self.domain[0], scale=self.domain[1] - self.domain[0]
        )

    def __call__(self, points: np.ndarray) -> np.ndarray:
        num_dat = points.shape[0]
        return np.full((num_dat,), self.normalization_constant)

    def sample(
        self,
        n_sample: IntLike,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return self.random_variable.rvs(
            size=(n_sample, self.input_dim), random_state=rng
        )
