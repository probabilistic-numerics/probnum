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
    n_nodes
        The number of nodes to be designed.
    measure
        The integration measure.

    """

    def __init__(self, n_nodes: IntLike, measure: IntegrationMeasure) -> None:
        super().__init__(measure=measure, n_nodes=n_nodes)

    @property
    def requires_rng(self) -> bool:
        return True

    def __call__(self, rng: Optional[np.random.Generator]) -> np.ndarray:
        return self.measure.sample(self.n_nodes, rng=rng)
