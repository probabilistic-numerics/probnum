"""Belief update for conjugate Bayesian quadrature."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from probnum.quad.kernel_embeddings import KernelEmbedding
from probnum.quad.solvers._bq_state import BQState
from probnum.randprocs.kernels import Kernel
from probnum.randvars import Normal
from probnum.typing import FloatLike

from ._belief_update import BQBeliefUpdate


# pylint: disable=too-few-public-methods
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

    # pylint: disable=too-many-locals
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
        gram_cho_factor = self.compute_gram_cho_factor(gram)

        # Estimate scaling parameter
        new_scale_sq = self._estimate_scale(fun_evals, gram_cho_factor, bq_state)

        # Integral mean and variance
        weights = self.gram_cho_solve(gram_cho_factor, kernel_means)
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
            gram_cho_factor=gram_cho_factor,
            kernel_means=kernel_means,
        )

        return new_belief, new_state

    # pylint: disable=no-self-use
    def _estimate_kernel(self, kernel: Kernel) -> Tuple[Kernel, bool]:
        """Estimate the intrinsic kernel parameters. That is, all parameters except the
        scale."""
        new_kernel = kernel
        kernel_was_updated = False
        return new_kernel, kernel_was_updated

    def _estimate_scale(
        self,
        fun_evals: np.ndarray,
        gram_cho_factor: Tuple[np.ndarray, bool],
        bq_state: BQState,
    ) -> FloatLike:
        """Estimate the scale parameter."""
        if self.scale_estimation is None:
            new_scale_sq = bq_state.scale_sq
        elif self.scale_estimation == "mle":
            new_scale_sq = (
                fun_evals
                @ self.gram_cho_solve(gram_cho_factor, fun_evals)
                / fun_evals.shape[0]
            )
        else:
            raise ValueError(f"Scale estimation ({self.scale_estimation}) is unknown.")
        return new_scale_sq

    @staticmethod
    def predict_integrand(
        x: np.ndarray, bq_state: BQState
    ) -> Tuple[np.ndarray, np.ndarray]:

        predictive_mean = np.zeros(x.shape[0])  # zero mean prior
        predictive_var = bq_state.kernel(x, x)

        nevals = bq_state.fun_evals.shape[0]
        if nevals != 0:
            kXx = bq_state.kernel.matrix(bq_state.nodes, x)
            weights = BQStandardBeliefUpdate.gram_cho_solve(
                bq_state.gram_cho_factor, kXx
            )

            # values (with zero mean prior at evals)
            predictive_mean += weights.T @ (bq_state.fun_evals - np.zeros(nevals))

            # variances
            predictive_var -= np.sum(weights * kXx, axis=0)

        return predictive_mean, bq_state.scale_sq * predictive_var
