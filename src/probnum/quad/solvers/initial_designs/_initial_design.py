"""Abstract base class of an initial design for Bayesian quadrature."""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure
from probnum.typing import IntLike


# pylint: disable=too-few-public-methods
class InitialDesign(abc.ABC):
    """An abstract class for an initial design for Bayesian quadrature.

    Parameters
    ----------
    num_nodes
        The number of nodes to be designed.
    measure
        The integration measure.

    """

    def __init__(self, num_nodes: IntLike, measure: IntegrationMeasure) -> None:
        self.num_nodes = int(num_nodes)
        self.measure = measure

    @abc.abstractmethod
    def __call__(self, rng: Optional[np.random.Generator]) -> np.ndarray:
        """Get the initial nodes.

        Parameters
        ----------
        rng
            A random number generator.

        Returns
        -------
        nodes :
            *shape=(num_nodes, input_dim)* -- Initial design nodes.
        """
        raise NotImplementedError
