"""Latin hypercube initial design for Bayesian quadrature."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import qmc

from probnum.quad.integration_measures import IntegrationMeasure
from probnum.typing import IntLike

from ._initial_design import InitialDesign


# pylint: disable=too-few-public-methods
class LatinDesign(InitialDesign):
    """Initial design for Bayesian quadrature that samples from a Latin hypercube. [1]_

    Parameters
    ----------
    n_nodes
        The number of nodes to be designed.
    measure
        The integration measure.

    References
    ----------
    .. [1] Mckay et al., A Comparison of Three Methods for Selecting Values of Input
       Variables in the Analysis of Output from a Computer Code. Technometrics, 1979.

    """

    def __init__(self, n_nodes: IntLike, measure: IntegrationMeasure) -> None:
        if np.Inf in np.hstack([abs(measure.domain[0]), abs(measure.domain[1])]):
            raise ValueError(
                "Latin hypercube samples require a finite domain. "
                "At least one dimension seems to be unbounded."
            )

        super().__init__(measure=measure, n_nodes=n_nodes)

    @property
    def requires_rng(self) -> bool:
        return True

    def __call__(self, rng: Optional[np.random.Generator]) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.measure.input_dim, seed=rng)
        sample = sampler.random(n=self.n_nodes)
        return qmc.scale(sample, self.measure.domain[0], self.measure.domain[1])
