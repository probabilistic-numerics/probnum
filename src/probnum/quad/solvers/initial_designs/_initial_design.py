"""Abstract base class of an initial design for Bayesian quadrature."""

from __future__ import annotations

import abc

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure
from probnum.typing import IntLike


# pylint: disable=too-few-public-methods
class InitialDesign(abc.ABC):
    """An abstract class for an initial design for Bayesian quadrature.

    Parameters
    ----------
    measure
        The integration measure.
    num_nodes
        The number of nodes to be designed.
    """

    def __init__(self, measure: IntegrationMeasure, num_nodes: IntLike) -> None:
        self.measure = measure
        self.num_nodes = int(num_nodes)

    @abc.abstractmethod
    def __call__(self) -> np.ndarray:
        """Get the initial nodes.

        Returns
        -------
        nodes :
            *shape=(num_nodes, input_dim)* -- Initial design nodes.
        """
        raise NotImplementedError
