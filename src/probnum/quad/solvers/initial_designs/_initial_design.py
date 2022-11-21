"""Abstract base class of an initial design for Bayesian quadrature."""

import abc

import numpy as np

from probnum.quad.integration_measures import IntegrationMeasure


# pylint: disable=too-few-public-methods
class InitialDesign(abc.ABC):
    """An abstract class for an initial design for Bayesian quadrature.

    Parameters
    ----------
    measure
        The integration measure.
    """

    def __init__(self, measure: IntegrationMeasure) -> None:
        self.measure = measure

    @abc.abstractmethod
    def __call__(self, num_nodes: int) -> np.ndarray:
        """Get the initial nodes.

        Parameters
        ----------
        num_nodes
            The number of nodes to be designed.
        Returns
        -------
        nodes :
            *shape=(num_nodes, input_dim)* -- Initial design nodes.
        """
        raise NotImplementedError
