"""Belief updates for Bayesian quadrature."""

import abc
from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.quad.solvers.bq_state import BQState
from probnum.randvars import Normal

# pylint: disable=too-few-public-methods, too-many-locals


class BQBeliefUpdate(abc.ABC):
    """Abstract class for the inference scheme."""

    def __call__(self, *args, **kwargs):
        pass


class BQStandardBeliefUpdate(BQBeliefUpdate):
    """Updates integral belief and state using standard Bayesian quadrature based on
    standard Gaussian process inference."""

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
            Gaussian integral belief after conditioning on the new nodes and evaluations.
        updated_state :
            Updated version of ``bq_state`` that contains all updated quantities.
        """

        # Update nodes and function evaluations
        old_nodes = bq_state.nodes

        nodes = np.concatenate((bq_state.nodes, new_nodes), axis=0)
        fun_evals = np.append(bq_state.fun_evals, new_fun_evals)

        # kernel quantities
        if old_nodes.size == 0:
            gram = bq_state.kernel.matrix(new_nodes)
            kernel_means = bq_state.kernel_embedding.kernel_mean(new_nodes)
        else:
            gram_new_new = bq_state.kernel.matrix(new_nodes)
            gram_old_new = bq_state.kernel.matrix(new_nodes, old_nodes)
            gram = np.hstack(
                (
                    np.vstack((bq_state.gram, gram_old_new)),
                    np.vstack((gram_old_new.T, gram_new_new)),
                )
            )
            kernel_means = np.concatenate(
                (
                    bq_state.kernel_means,
                    bq_state.kernel_embedding.kernel_mean(new_nodes),
                )
            )

        initial_integral_variance = bq_state.kernel_embedding.kernel_variance()
        weights = self._solve_gram(gram, kernel_means)

        # integral mean and variance
        integral_mean = weights @ fun_evals
        integral_variance = initial_integral_variance - weights @ kernel_means

        updated_belief = Normal(integral_mean, integral_variance)
        updated_state = BQState.from_new_data(
            nodes=nodes,
            fun_evals=fun_evals,
            integral_belief=updated_belief,
            prev_state=bq_state,
            gram=gram,
            kernel_means=kernel_means,
        )

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
