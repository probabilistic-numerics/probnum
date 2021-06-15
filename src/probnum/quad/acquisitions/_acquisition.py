"""Abstract base class for BQ acquistion function."""


import abc
from typing import Tuple, Union

import numpy as np

from probnum.quad.bq_methods.bq_state import BQState


class Acquisition(abc.ABC):
    """Abstract base class for BQ acquisition functions.

    Parameters
    ----------
    bq_state :
        The state of the Bayesian quadrature loop.
    """

    def __init__(self, bq_state: BQState) -> None:
        self.bq_state = bq_state

    def __call__(self, nodes: np.ndarray) -> Union[float, Tuple[float, np.ndarray]]:
        """Evaluate the acquisition function.

        Parameters
        ----------
        nodes :
            *shape=(batch_size, dim)* -- Locations for which to query the acquisition function.

        Returns
        -------
        evaluation, [gradient] :
            Evaluation and, if available, the gradient.
        """
        if self.has_gradients:
            return self._evaluate_with_gradients(nodes)
        else:
            return self._evaluate(nodes)

    @abc.abstractmethod
    def _evaluate(self, nodes: np.ndarray) -> float:
        pass

    def _evaluate_with_gradients(self, nodes: np.ndarray) -> Tuple[float, np.ndarray]:
        raise NotImplementedError(
            "No gradients available for this acquisition function"
        )

    @property
    @abc.abstractmethod
    def has_gradients(self) -> bool:
        pass

    def sample(self) -> np.ndarray:
        """Sample from a density that is proportional to the acquisition function.

        Returns
        -------
        nodes:
            *shape=(batch_size, dim)* -- Sampled nodes.
        """
        raise NotImplementedError(
            "Sampling is not available for this acquisition function"
        )
