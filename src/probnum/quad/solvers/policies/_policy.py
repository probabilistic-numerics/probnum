"""Abstract base class for BQ policies."""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from probnum.quad.solvers._bq_state import BQState
from probnum.typing import IntLike

# pylint: disable=too-few-public-methods, fixme


class Policy(abc.ABC):
    """An abstract class for a policy that acquires nodes for Bayesian quadrature.

    Parameters
    ----------
    batch_size
        Size of batch of nodes when calling the policy once.
    """

    def __init__(self, batch_size: IntLike) -> None:
        self.batch_size = int(batch_size)

    @property
    @abc.abstractmethod
    def requires_rng(self) -> bool:
        """Whether the policy requires a random number generator when called."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self, bq_state: BQState, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        """Find nodes according to the policy.

        Parameters
        ----------
        bq_state
            State of the BQ belief.
        rng
            A random number generator.

        Returns
        -------
        nodes :
            *shape=(batch_size, input_dim)* -- Nodes found according to the policy.
        """
        raise NotImplementedError
