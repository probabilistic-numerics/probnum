"""Initial design that samples from the integration measure."""

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure
from ._initial_design import InitialDesign


class BMCDesign(InitialDesign):
    """Initial design for Bayesian quadrature that samples from the integration measure.

    Parameters
    ----------
    measure
        The integration measure.
     rng
        A random number generator.
   """

    def __init__(self, measure: IntegrationMeasure, rng: np.random.Generator = np.random.default_rng()) -> None:
        # Todo: check if sampling from measure is possible.
        super().__init__(measure=measure)
        self.rng = rng

    def __call__(self, num_nodes: int) -> np.ndarray:
        return self.measure.sample(num_nodes, rng=self.rng)
