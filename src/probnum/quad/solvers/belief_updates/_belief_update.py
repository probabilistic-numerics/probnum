"""Base class of belief update for Bayesian quadrature."""

from __future__ import annotations

import abc
from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.quad.solvers._bq_state import BQState
from probnum.randvars import Normal
from probnum.typing import FloatLike


# pylint: disable=too-few-public-methods
class BQBeliefUpdate(abc.ABC):
    """Abstract class for the inference scheme.

    Parameters
    ----------
    jitter
        Non-negative jitter to numerically stabilise kernel matrix inversion.
    """

    def __init__(self, jitter: FloatLike) -> None:
        if jitter < 0:
            raise ValueError(f"Jitter ({jitter}) must be non-negative.")
        self.jitter = float(jitter)

    @abc.abstractmethod
    def __call__(
        self,
        bq_state: BQState,
        new_nodes: np.ndarray,
        new_fun_evals: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[Normal, BQState]:
        """Updates integral belief and BQ state according to the new data given.

        Parameters
        ----------
        bq_state :
            Current state of the Bayesian quadrature loop.
        new_nodes :
            *shape=(n_eval_new, input_dim)* -- New nodes that have been added.
        new_fun_evals :
            *shape=(n_eval_new,)* -- Function evaluations at the given node.

        Returns
        -------
        updated_belief :
            Gaussian integral belief after conditioning on the new nodes and
            evaluations.
        updated_state :
            Updated version of ``bq_state`` that contains all updated quantities.
        """
        raise NotImplementedError

    def compute_gram_cho_factor(self, gram: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Compute the Cholesky decomposition of a positive-definite Gram matrix for use
        in scipy.linalg.cho_solve

        .. warning::
            Uses scipy.linalg.cho_factor. The returned matrix is only to be used in
            scipy.linalg.cho_solve.

        Parameters
        ----------
        gram
            symmetric pos. def. kernel Gram matrix :math:`K`, shape (nevals, nevals)

        Returns
        -------
        gram_cho_factor :
            The upper triangular Cholesky decomposition of the Gram matrix. Other
            parts of the matrix contain random data. A boolean that indicates whether
            the matrix is lower triangular (always False but needed for scipy).
        """
        return cho_factor(gram + self.jitter * np.eye(gram.shape[0]))

    @staticmethod
    def gram_cho_solve(
        gram_cho_factor: Tuple[np.ndarray, bool], z: np.ndarray
    ) -> np.ndarray:
        """Wrapper for scipy.linalg.cho_solve. Meant to be used for linear systems of
        the gram matrix. Requires the solution of scipy.linalg.cho_factor as input.

        Parameters
        ----------
        gram_cho_factor
            The return object of compute_gram_cho_factor.
        z
            An array of appropriate shape.

        Returns
        -------
        solution :
            The solution ``x`` to the linear system ``gram x = z``.

        """
        return cho_solve(gram_cho_factor, z)

    @staticmethod
    @abc.abstractmethod
    def predict_integrand(
        x: np.ndarray, bq_state: BQState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean and variances of the integrand at given nodes.

        Parameters
        ----------
        x
            *shape=(n_nodes, input_dim)* -- The nodes where to predict.
        bq_state
            The BQ state.

        Returns
        -------
        mean_prediction :
            *shape=(n_nodes,)* -- The means of the predictions.
        var_predictions :
            *shape=(n_nodes,)* -- The variances of the predictions.

        """
        raise NotImplementedError
