"""Belief updates for Bayesian quadrature."""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.backend.typing import FloatLike
from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.quad.solvers._bq_state import BQState
from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal

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

    def _compute_gram_cho_factor(self, gram: np.ndarray) -> np.ndarray:
        """Compute the Cholesky decomposition of a positive-definite Gram matrix for use
        in scipy.linalg.cho_solve.

        .. warning::
            Uses scipy.linalg.cho_factor. The returned matrix is only to be used in
            scipy.linalg.cho_solve.

        Parameters
        ----------
        gram :
            symmetric pos. def. kernel Gram matrix :math:`K`, shape (nevals, nevals)

        Returns
        -------
        gram_cho_factor :
            The upper triangular Cholesky decomposition of the Gram matrix. Other
            parts of the matrix contain random data.
        """
        return cho_factor(gram + self.jitter * np.eye(gram.shape[0]))

    # pylint: disable=no-self-use
    def _gram_cho_solve(self, gram_cho_factor: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Wrapper for scipy.linalg.cho_solve.

        Meant to be used for linear systems of the gram matrix. Requires the solution of
        scipy.linalg.cho_factor as input.
        """
        return cho_solve(gram_cho_factor, z)


class BQStandardBeliefUpdate(BQBeliefUpdate):
    """Updates integral belief and state using standard Bayesian quadrature based on
    standard Gaussian process inference.

    Parameters
    ----------
    jitter
        Non-negative jitter to numerically stabilise kernel matrix inversion.
    scale_estimation
        Estimation method to use to compute the scale parameter.
    """

    def __init__(self, jitter: FloatLike, scale_estimation: Optional[str]) -> None:
        super().__init__(jitter=jitter)
        self.scale_estimation = scale_estimation

    def __call__(
        self,
        bq_state: BQState,
        new_nodes: np.ndarray,
        new_fun_evals: np.ndarray,
        *args,
        **kwargs,
    ) -> Tuple[Normal, BQState]:

        # Update nodes and function evaluations
        nodes = np.concatenate((bq_state.nodes, new_nodes), axis=0)
        fun_evals = np.append(bq_state.fun_evals, new_fun_evals)

        # Estimate intrinsic kernel parameters
        new_kernel, kernel_was_updated = self._estimate_kernel(bq_state.kernel)
        new_kernel_embedding = KernelEmbedding(new_kernel, bq_state.measure)

        # Update gram matrix and kernel mean vector. Recompute everything from
        # scratch if the kernel was updated or if these are the first nodes.
        if kernel_was_updated or bq_state.nodes.size == 0:
            gram = new_kernel.matrix(nodes)
            kernel_means = new_kernel_embedding.kernel_mean(nodes)
        else:
            gram_new_new = new_kernel.matrix(new_nodes)
            gram_old_new = new_kernel.matrix(new_nodes, bq_state.nodes)
            gram = np.hstack(
                (
                    np.vstack((bq_state.gram, gram_old_new)),
                    np.vstack((gram_old_new.T, gram_new_new)),
                )
            )
            kernel_means = np.concatenate(
                (
                    bq_state.kernel_means,
                    new_kernel_embedding.kernel_mean(new_nodes),
                )
            )

        # Cholesky factorisation of the Gram matrix
        gram_cho_factor = self._compute_gram_cho_factor(gram)

        # Estimate scaling parameter
        new_scale_sq = self._estimate_scale(fun_evals, gram_cho_factor, bq_state)

        # Integral mean and variance
        weights = self._gram_cho_solve(gram_cho_factor, kernel_means)
        integral_mean = weights @ fun_evals
        initial_integral_variance = new_kernel_embedding.kernel_variance()
        integral_variance = new_scale_sq * (
            initial_integral_variance - weights @ kernel_means
        )

        new_belief = Normal(integral_mean, integral_variance)
        new_state = BQState.from_new_data(
            kernel=new_kernel,
            scale_sq=new_scale_sq,
            nodes=nodes,
            fun_evals=fun_evals,
            integral_belief=new_belief,
            prev_state=bq_state,
            gram=gram,
            kernel_means=kernel_means,
        )

        return new_belief, new_state

    # pylint: disable=no-self-use
    def _estimate_kernel(self, kernel: Kernel) -> Tuple[Kernel, bool]:
        """Estimate the intrinsic kernel parameters.

        That is, all parameters except the scale.
        """
        new_kernel = kernel
        kernel_was_updated = False
        return new_kernel, kernel_was_updated

    def _estimate_scale(
        self, fun_evals: np.ndarray, gram_cho_factor: np.ndarray, bq_state: BQState
    ) -> FloatLike:
        """Estimate the scale parameter."""
        if self.scale_estimation is None:
            new_scale_sq = bq_state.scale_sq
        elif self.scale_estimation == "mle":
            new_scale_sq = (
                fun_evals
                @ self._gram_cho_solve(gram_cho_factor, fun_evals)
                / fun_evals.shape[0]
            )
        else:
            raise ValueError(f"Scale estimation ({self.scale_estimation}) is unknown.")
        return new_scale_sq
