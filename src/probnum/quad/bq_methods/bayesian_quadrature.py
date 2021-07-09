"""Probabilistic numerical methods for solving integrals."""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.kernels import ExpQuad, Kernel
from probnum.randvars import Normal

from .._integration_measures import IntegrationMeasure
from ..kernel_embeddings import KernelEmbedding
from ..policies import sample_from_measure


class BayesianQuadrature:
    r"""A base class for Bayesian quadrature.

    Bayesian quadrature solves integrals of the form

    .. math:: F = \int_\Omega f(x) d \mu(x).

    Parameters
    ----------
    kernel:
        The kernel used for the GP model.
    policy:
        The policy for acquiring nodes for function evaluations.
    """

    def __init__(self, kernel: Kernel, policy: Callable) -> None:
        self.kernel = kernel
        self.policy = policy

    @classmethod
    def from_interface(
        cls,
        input_dim: int,
        kernel: Optional[Kernel] = None,
        method: str = "vanilla",
        policy: str = "bmc",
        rng: np.random.Generator = None,
    ) -> "BayesianQuadrature":
        if method != "vanilla":
            raise NotImplementedError
        if kernel is None:
            kernel = ExpQuad(input_dim=input_dim)
        if policy == "bmc":
            if rng is None:
                errormsg = (
                    "Policy 'bmc' relies on random sampling, "
                    "thus requires a random number generator ('rng')."
                )
                raise ValueError(errormsg)

            def policy(nevals, measure):
                return sample_from_measure(rng=rng, nevals=nevals, measure=measure)

        else:
            raise NotImplementedError(
                "Policies other than random sampling are not available at the moment."
            )
        return cls(kernel=kernel, policy=policy)

    def integrate(
        self, fun: Callable, measure: IntegrationMeasure, nevals: int
    ) -> Tuple[Normal, Dict]:
        r"""Integrate the function ``fun``.

        Parameters
        ----------
        fun:
            The integrand function :math:`f`.
        measure :
            An integration measure :math:`\mu`.
        nevals :
            Number of function evaluations.

        Returns
        -------
        F :
            The integral of ``fun`` against ``measure``.
        info :
            Information on the performance of the method.
        """

        # Acquisition policy
        nodes = self.policy(nevals, measure)
        fun_evals = fun(nodes)

        # compute integral mean and variance
        # Define kernel embedding
        kernel_embedding = KernelEmbedding(self.kernel, measure)
        gram = self.kernel(nodes, nodes)
        kernel_mean = kernel_embedding.kernel_mean(nodes)
        initial_error = kernel_embedding.kernel_variance()

        weights = self._solve_gram(gram, kernel_mean)

        integral_mean = np.squeeze(weights.T @ fun_evals)
        integral_variance = initial_error - weights.T @ kernel_mean

        integral = Normal(integral_mean, integral_variance)

        # Information on result
        info = {"model_fit_diagnostic": None}

        return integral, info

    # The following functions are here for the minimal version
    # and shall be factored out once BQ is expanded.
    # 1. acquisition policy
    # 2. GP inference

    @staticmethod
    def _solve_gram(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Solve the linear system of the form.

        .. math:: Kx=b.

        Parameters
        ----------
        gram :
            *shape=(nevals, nevals)* -- Symmetric pos. def. kernel Gram matrix :math:`K`.
        rhs :
            *shape=(nevals, ...)* -- Right-hand-side :math:`b`, matrix or vector.

        Returns
        -------
        x:
            The solution to the linear system :math:`K x = b`.
        """
        jitter = 1.0e-6
        chol_gram = cho_factor(gram + jitter * np.eye(gram.shape[0]))
        return cho_solve(chol_gram, rhs)
