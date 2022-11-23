"""Initial design that samples from the integration measure."""

from __future__ import annotations

from typing import Optional

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure
from probnum.typing import IntLike

from ._initial_design import InitialDesign


# pylint: disable=too-few-public-methods
class MCDesign(InitialDesign):
    """Initial design for Bayesian quadrature that samples from the integration measure.

    Parameters
    ----------
    num_nodes
        The number of nodes to be designed.
    measure
        The integration measure.

    """

    def __init__(self, num_nodes: IntLike, measure: IntegrationMeasure) -> None:
        super().__init__(measure=measure, num_nodes=num_nodes)

    def __call__(self, rng: Optional[np.random.Generator]) -> np.ndarray:
        return self.measure.sample(self.num_nodes, rng=rng)
