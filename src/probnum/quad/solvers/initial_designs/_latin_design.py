"""Latin hypercube initial design."""

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure

from ._initial_design import InitialDesign


# pylint: disable=too-few-public-methods
class LatinDesign(InitialDesign):
    """Initial design for Bayesian quadrature that samples from a latin hypercube.

    Parameters
    ----------
    measure
        The integration measure.
     rng
        A random number generator.
    """

    def __init__(
        self,
        measure: IntegrationMeasure,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        # Todo: measure domain needs to be cube
        super().__init__(measure=measure)
        self.rng = rng

    def __call__(self, num_nodes: int) -> np.ndarray:
        pass
