"""Abstract base class for BQ acquisition policies."""

import abc
from typing import Optional, Tuple, Union

import numpy as np

from probnum.quad.bq_methods import BQState
from probnum.randvars import Normal
from probnum.type import FloatArgType, IntArgType, RandomStateArgType


class Policy(abc.ABC):
    """An abstract class for acquisition functions for Bayesian quadrature.

    Parameters
    ----------
    batch_size :
        Size of batch that is found in every iteration of calling the policy.
    bq_state :
        Current state of the BQ method.
    mode:
        Whether the policy relies on sampling or optimization.

        =======================  ============
         Sample from objective   ``sample``
         Optimize objective      ``optimize``
        =======================  ============
    """

    def __init__(self, batch_size: int, bq_state: BQState, mode: str) -> None:
        self.batch_size = batch_size
        self.bq_state = bq_state
        self.mode = mode

    def __call__(self) -> np.ndarray:
        """Find nodes according to the acquisition policy.

        The nodes found are the argmin of the given acquistion function

        Returns
        -------
        nodes :
            *shape=(batch_size, dim)* -- Nodes found according to the policy.
        """
        if self.mode == "sample":
            nodes = self._sample_from_objective()
        elif self.mode == "optimize":
            nodes = self._optimize_objective()
        else:
            raise ValueError(f"The option mode='{self.mode}' is not available.")
        return nodes

    def objective(self, nodes: np.ndarray) -> int:
        """The acquisition function.

        Parameters
        ----------
        nodes:
            *shape=(n_eval, dim)* -- Nodes

        Returns
        -------
        acq:
            value of the acquisition function for input(s) ``nodes``.
        """
        raise NotImplementedError

    def _sample_from_objective(self) -> np.ndarray:
        """Sample from the objective.

        Assumes that the objective is a positive function with a finite integral over the
        integration domain.

        Returns
        -------
        nodes :
            *shape=(batch_size, dim)* -- Nodes found by optimizing the objective.
        """
        raise NotImplementedError

    def _optimize_objective(self) -> np.ndarray:
        """Find the argmax of the objective.

        Returns
        -------
        nodes :
            *shape=(batch_size, dim)* -- Nodes found by optimizing the objective.
        """
        raise NotImplementedError
