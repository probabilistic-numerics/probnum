"""Belief updates for Bayesian quadrature."""

import abc
from typing import Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.quad.solvers.bq_state import BQState
from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal
from probnum.typing import FloatLike

# pylint: disable=too-few-public-methods, too-many-locals


class BQBeliefUpdate(abc.ABC):
    """Abstract class for the inference scheme.

    Parameters
    ----------
    jitter
        Non-negative jitter to numerically stabilise kernel matrix inversion.
    """

    def __init__(self, jitter: FloatLike) -> None:
        if jitter < 0:
            raise ValueError("Jitter must be non-negative.")
        self.jitter = jitter

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

    def _gram_cholesky_decomposition(self, gram: np.ndarray) -> np.ndarray:
        """Find the Cholesky decomposition of a positive-definite Gram matrix.

        Parameters
        ----------
        gram :
            symmetric pos. def. kernel Gram matrix :math:`K`, shape (nevals, nevals)

        Returns
        -------
        chol_gram :
            The Cholesky decomposition of the Gram matrix.
        """
        chol_gram = cho_factor(gram + self.jitter * np.eye(gram.shape[0]))
        return chol_gram


class BQStandardBeliefUpdate(BQBeliefUpdate):
    """Updates integral belief and state using standard Bayesian quadrature based on
    standard Gaussian process inference.

    Parameters
    ----------
    jitter
        Non-negative jitter to numerically stabilise kernel matrix inversion.
    scale_estimator
        Estimator to use to compute the scale parameter.
    """

    def __init__(self, jitter: FloatLike, scale_estimator: str) -> None:
        super().__init__(jitter=jitter)
        self.scale_estimator = scale_estimator

    def __call__(
        self,
        bq_state: BQState,
        new_nodes: np.ndarray,
        new_fun_evals: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[Normal, BQState]:

        # Update nodes and function evaluations
        old_nodes = bq_state.nodes

        nodes = np.concatenate((bq_state.nodes, new_nodes), axis=0)
        fun_evals = np.append(bq_state.fun_evals, new_fun_evals)

        # Update intrinsic kernel parameters
        updated_kernel, kernel_was_updated = self._update_kernel_params(bq_state.kernel)
        updated_kernel_embedding = KernelEmbedding(updated_kernel, bq_state.measure)

        # Update gram matrix and kernel mean vector. Recompute everything from
        # scratch if the kernel was updated or if these are the first nodes.
        if kernel_was_updated or old_nodes.size == 0:
            gram = updated_kernel.matrix(nodes)
            kernel_means = updated_kernel_embedding.kernel_mean(nodes)
        else:
            gram_new_new = updated_kernel.matrix(new_nodes)
            gram_old_new = updated_kernel.matrix(new_nodes, old_nodes)
            gram = np.hstack(
                (
                    np.vstack((bq_state.gram, gram_old_new)),
                    np.vstack((gram_old_new.T, gram_new_new)),
                )
            )
            kernel_means = np.concatenate(
                (
                    bq_state.kernel_means,
                    updated_kernel_embedding.kernel_mean(new_nodes),
                )
            )

        chol_gram = self._gram_cholesky_decomposition(gram)

        # Update scaling parameter
        updated_scale_sq = self._update_scale(fun_evals, chol_gram)

        initial_integral_variance = updated_kernel_embedding.kernel_variance()
        weights = self._solve_gram(gram, kernel_means)

        # integral mean and variance
        integral_mean = weights @ fun_evals
        integral_variance = updated_scale_sq * (
            initial_integral_variance - weights @ kernel_means
        )

        updated_belief = Normal(integral_mean, integral_variance)
        updated_state = BQState.from_new_data(
            kernel=updated_kernel,
            scale_sq=updated_scale_sq,
            nodes=nodes,
            fun_evals=fun_evals,
            integral_belief=updated_belief,
            prev_state=bq_state,
            gram=gram,
            kernel_means=kernel_means,
        )

        return updated_belief, updated_state

    def _update_kernel_params(self, kernel: Kernel) -> Tuple[Kernel, bool]:
        """Update the intrinsic kernel parameters. That is, all parameters except the
        scale."""
        updated_kernel = kernel
        kernel_was_updated = False
        return updated_kernel, kernel_was_updated

    def _update_scale(self, fun_evals: np.ndarray, chol_gram: np.ndarray) -> FloatLike:
        """Update the scale parameter."""
        if self.scale_estimator == "mle":
            updated_scale_sq = (
                fun_evals @ cho_solve(chol_gram, fun_evals) / fun_evals.shape[0]
            )
        else:
            updated_scale_sq = 1.0
        return updated_scale_sq
