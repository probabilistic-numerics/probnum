"""Abstract base class for BQ acquisition functions."""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import numpy as np

from probnum.quad.solvers._bq_state import BQState
from probnum.quad.solvers.belief_updates import BQBeliefUpdate


class AcquisitionFunction(abc.ABC):
    """An abstract class for an acquisition function for Bayesian quadrature."""

    @property
    @abc.abstractmethod
    def has_gradients(self) -> bool:
        """Whether the acquisition function exposes gradients."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self, x: np.ndarray, bq_state: BQState, belief_update: BQBeliefUpdate
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Evaluates the acquisition function and its gradients.

        Parameters
        ----------
        x
            *shape=(batch_size, input_dim)* -- The nodes where to evaluate the
            acquisition function at.
        bq_state
            State of the BQ belief.
        belief_update
            The belief update.

        Returns
        -------
        nodes :
            *shape=(batch_size, )* -- The acquisition values.
            *shape=(batch_size, input_dim)* -- The corresponding gradients (optional).
        """
        raise NotImplementedError
