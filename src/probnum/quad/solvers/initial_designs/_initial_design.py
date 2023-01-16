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
    n_nodes
        The number of nodes to be designed.
    measure
        The integration measure.

    """

    def __init__(self, n_nodes: IntLike, measure: IntegrationMeasure) -> None:
        self.n_nodes = int(n_nodes)
        self.measure = measure

    @property
    @abc.abstractmethod
    def requires_rng(self) -> bool:
        """Whether the initial design requires a random number generator when called."""
        raise NotImplementedError

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
            *shape=(n_nodes, input_dim)* -- Initial design nodes.
        """
        raise NotImplementedError
