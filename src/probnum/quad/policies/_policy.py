"""Abstract base class for BQ acquisition policies."""

import abc
from typing import Callable

import numpy as np

from probnum.quad.acquisitions import Acquisition
from probnum.quad.bq_methods.bq_state import BQState


class Policy(abc.ABC):
    """An abstract class for acquisition functions for Bayesian quadrature.

    Parameters
    ----------
    batch_size :
        Size of batch that is found in every iteration of calling the policy.
    """

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def __call__(self, bq_state: BQState) -> np.ndarray:
        """Find nodes according to the acquisition policy.

        Returns
        -------
        nodes :
            *shape=(batch_size, input_dim)* -- Nodes found according to the policy.
        """
        # raise NotImplementedError
        return None


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
        super().__init__(batch_size=batch_size)
        self.acquisition = acquisition

    def __call__(self, bq_state: BQState) -> np.ndarray:
        """Find nodes according to the acquisition policy.

        The nodes found are the :math:`\\argmax` of the given acquisition function.

        Returns
        -------
        nodes :
            *shape=(batch_size, input_dim)* -- Nodes found according to the policy.
        """
        # TODO: Here goes the optimization of the acquisition function.
        raise NotImplementedError


class RandomPolicy(Policy):
    """Random sampling from an objective.

    Parameters
    ----------
    sample :
        Objective to sample from. Needs to have a method ``sample``.
    batch_size :
        Size of batch that is found in every iteration of calling the policy.
    """

    def __init__(
        self,
        sample: Callable,
        batch_size: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(batch_size=batch_size)
        self.sample = sample
        self.rng = rng

    def __call__(self, bq_state: BQState) -> np.ndarray:
        """Sample nodes.

        Parameters
        ----------
        bq_state :
            Current state of the BQ method.

        Returns
        -------
        nodes :
            *shape=(batch_size, input_dim)* -- Nodes found according to the policy.
        """
        return self.sample(self.batch_size, rng=self.rng)
