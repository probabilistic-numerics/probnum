"""Abstract base class for BQ acquisition policies."""

import abc
from typing import Callable

import numpy as np

from probnum.quad.solvers.bq_state import BQState

# pylint: disable=too-few-public-methods, fixme


class Policy(abc.ABC):
    """An abstract class for a policy that acquires nodes for Bayesian quadrature.

    Parameters
    ----------
    batch_size :
        Size of batch of nodes when calling the policy once.
    """

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def __call__(self, bq_state: BQState) -> np.ndarray:
        """Find nodes according to the policy.

        Parameters
        ----------
        bq_state :
            Current state of the BQ method.

        Returns
        -------
        nodes :
            *shape=(batch_size, input_dim)* -- Nodes found according to the policy.
        """
        raise NotImplementedError


class RandomPolicy(Policy):
    """Random sampling from an objective.

    Parameters
    ----------
    sample_func :
        The sample function. Needs to have the following interface:
        `sample_func(batch_size: int, rng: np.random.Generator)` and return an array of shape (batch_size, n_dim).
    batch_size :
        Size of batch of nodes when calling the policy once.
    """

    def __init__(
        self,
        sample_func: Callable,
        batch_size: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(batch_size=batch_size)
        self.sample_func = sample_func
        self.rng = rng

    def __call__(self, bq_state: BQState) -> np.ndarray:
        """Find nodes according to the random policy.

        Parameters
        ----------
        bq_state :
            Current state of the BQ method.

        Returns
        -------
        nodes :
            *shape=(batch_size, input_dim)* -- Nodes found according to the policy.
        """
        return self.sample_func(self.batch_size, rng=self.rng)
