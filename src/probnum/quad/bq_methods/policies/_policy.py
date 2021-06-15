"""Abstract base class for BQ acquisition policies."""

import abc

import numpy as np

from ..acquisitions import Acquisition
from ..bq_state import BQState


class Policy(abc.ABC):
    """An abstract class for acquisition functions for Bayesian quadrature.

    Parameters
    ----------
    batch_size :
        Size of batch that is found in every iteration of calling the policy.
    is_deterministic :
        Is the policy a deterministic function of its arguments or stochastic (i.e.
        sampling-based)?
    """

    def __init__(self, batch_size: int, is_deterministic: bool) -> None:
        self.batch_size = batch_size
        self._is_deterministic = is_deterministic

    def __call__(self, bq_state: BQState) -> np.ndarray:
        """Find nodes according to the acquisition policy.

        The nodes found are the argmin of the given acquistion function

        Returns
        -------
        nodes :
            *shape=(batch_size, dim)* -- Nodes found according to the policy.
        """
        raise NotImplementedError

    @property
    def is_deterministic(self) -> bool:
        """Is the policy a deterministic function of its arguments or stochastic?"""
        return self._is_deterministic


class OptimalPolicy(Policy):
    """Optimal policy given an acquisition function.

    Parameters
    ----------
    acquisition :
        Acquisition function.
    batch_size :
        Size of batch that is found in every iteration of calling the policy.
    bq_state :
        Current state of the BQ method.
    """

    def __init__(self, acquisition: Acquisition, batch_size: int) -> None:
        self.acquisition = acquisition
        self.batch_size = batch_size

        super().__init__(batch_size=batch_size, is_deterministic=True)

    def __call__(self, bq_state: BQState) -> np.ndarray:
        """Find nodes according to the acquisition policy.

        The nodes found are the argmin of the given acquistion function

        Returns
        -------
        nodes :
            *shape=(batch_size, dim)* -- Nodes found according to the policy.
        """
        # TODO: Here goes the optimization of the acquisition function.
        raise NotImplementedError


class RandomPolicy(Policy):
    """Random sampling from an objective.

    Parameters
    ----------
    sampling_objective :
        Objective to sample from. Needs to have a method ``sample``.
    batch_size :
        Size of batch that is found in every iteration of calling the policy.
    """

    def __init__(self, sampling_objective, batch_size: int) -> None:
        self.sampling_objective = sampling_objective
        self.batch_size = batch_size

        super().__init__(batch_size=batch_size, is_deterministic=False)

    def __call__(self, bq_state: BQState) -> np.ndarray:
        """Sample nodes.

        Parameters
        ----------
        bq_state :
            Current state of the BQ method.

        Returns
        -------
        nodes :
            *shape=(batch_size, dim)* -- Nodes found according to the policy.
        """
        return self.sampling_objective.sample(self.batch_size)
