"""Abstract base class for BQ policies."""

import abc

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
            State of the BQ belief.
        Returns
        -------
        nodes :
            *shape=(batch_size, input_dim)* -- Nodes found according to the policy.
        """
        raise NotImplementedError
