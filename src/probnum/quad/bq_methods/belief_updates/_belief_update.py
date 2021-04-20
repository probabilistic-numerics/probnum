"""Belief updates for Bayesian quadrature."""

import abc
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.randvars import Normal


class BQBeliefUpdate(abc.ABC):
    def __init__(self):
        pass


class BQStandardBeliefUpdate(BQBeliefUpdate):
    def __init__(self):
        pass

    def __call__(
        self, fun, measure, kernel, integral_belief, new_nodes, new_fun_evals, bq_state
    ):
        # compute integral mean and variance
        # Define kernel embedding
        nodes = bq_state.nodes
        gram = kernel(nodes, nodes)
        kernel_mean = bq_state.kernel_embedding.kernel_mean(nodes)
        initial_error = bq_state.kernel_embedding.kernel_variance()

        weights = self._solve_gram(gram, kernel_mean)

        integral_mean = np.squeeze(weights.T @ bq_state.fun_evals)
        integral_variance = initial_error - weights.T @ kernel_mean

        updated_belief = Normal(integral_mean, integral_variance)
        updated_state = bq_state
        updated_state.integral_belief = updated_belief

        return updated_belief, updated_state

    @staticmethod
    def _solve_gram(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Solve the linear system of the form.

        .. math:: Kx=b,

        Parameters
        ----------
        gram :
            symmetric pos. def. kernel Gram matrix :math:`K`, shape (nevals, nevals)
        rhs :
            right-hand-side :math:`b`, matrix or vector, shape (nevals, ...)

        Returns
        -------
        x:
            The solution to the linear system :math:`K x = b`
        """
        jitter = 1.0e-6
        chol_gram = cho_factor(gram + jitter * np.eye(gram.shape[0]))
        return cho_solve(chol_gram, rhs)
