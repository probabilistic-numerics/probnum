"""Belief updates for Bayesian quadrature."""

import abc
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.quad.bq_methods.bq_state import BQState
from probnum.randvars import Normal


class BQBeliefUpdate(abc.ABC):
    """Abstract class for the inference scheme."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class BQStandardBeliefUpdate(BQBeliefUpdate):
    """Updates integral belief and state using standard Bayesian quadrature based on
    standard Gaussian process inference."""

    def __call__(
        self,
        bq_state: BQState,
        new_nodes: Optional[np.ndarray] = None,
        new_fun_evals: Optional[np.ndarray] = None,
    ):
        """Updates integral belief and BQ state according to the new data given.

        Parameters
        ----------
        bq_state :
            Current state of the Bayesian quadrature loop.
        new_nodes :
            New nodes that have been added.
        new_fun_evals :
            Function evaluations at the given node.

        Returns
        -------
        updated_belief :
            Gaussian integral belief after conditioning on the new nodes and evaluations.
        updated_state :
            Updated version of ``bq_state`` that contains all updated quantities.
        """

        # update nodes and function evaluations
        if new_nodes is None:
            nodes = bq_state.nodes
            fun_evals = bq_state.fun_evals
        else:
            if new_fun_evals is None:
                new_fun_evals = bq_state.fun(new_nodes)
            nodes = np.concatenate((bq_state.nodes, new_nodes), axis=0)
            fun_evals = np.append(bq_state.fun_evals, new_fun_evals)

        # kernel quantities
        gram = bq_state.kernel(nodes, nodes)
        kernel_mean = bq_state.kernel_embedding.kernel_mean(nodes)
        initial_error = bq_state.kernel_embedding.kernel_variance()
        weights = self._solve_gram(gram, kernel_mean)

        # integral mean and variance
        integral_mean = np.squeeze(weights.T @ fun_evals)
        integral_variance = initial_error - weights.T @ kernel_mean

        updated_belief = Normal(integral_mean, integral_variance)
        updated_state = BQState.from_new_data(
            nodes, fun_evals, updated_belief, bq_state
        )
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
