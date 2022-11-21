"""Latin hypercube initial design."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from probnum.quad.integration_measures import IntegrationMeasure
from probnum.typing import IntLike

from ._initial_design import InitialDesign


# pylint: disable=too-few-public-methods
class LatinDesign(InitialDesign):
    """Initial design for Bayesian quadrature that samples from a latin hypercube.

    Parameters
    ----------
    measure
        The integration measure.
    num_nodes
        The number of nodes to be designed.
    rng
        A random number generator.
    """

    def __init__(
        self,
        measure: IntegrationMeasure,
        num_nodes: IntLike,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        if np.Inf in [abs(measure.domain[0]), abs(measure.domain[1])]:
            raise ValueError(
                "Latin hypercube samples require a finite domain. "
                "At least one dimension seems to be unbounded."
            )

        super().__init__(measure=measure, num_nodes=num_nodes)
        self.rng = rng

    def __call__(self) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.measure.input_dim, seed=self.rng)
        sample = sampler.random(n=self.num_nodes)
        return qmc.scale(sample, self.measure.domain[0], self.measure.domain[1])
