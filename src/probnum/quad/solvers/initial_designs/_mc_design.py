"""Initial design that samples from the integration measure."""

from __future__ import annotations

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure
from probnum.typing import IntLike

from ._initial_design import InitialDesign


# pylint: disable=too-few-public-methods
class MCDesign(InitialDesign):
    """Initial design for Bayesian quadrature that samples from the integration measure.

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
        super().__init__(measure=measure, num_nodes=num_nodes)
        self.rng = rng

    def __call__(self) -> np.ndarray:
        return self.measure.sample(self.num_nodes, rng=self.rng)
