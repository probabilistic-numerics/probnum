"""Probabilistic numerical methods for solving integrals."""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.kernels import ExpQuad, Kernel
from probnum.random_variables import Normal

from .._integration_measures import IntegrationMeasure
from .._kernel_embeddings import get_kernel_embedding


class BayesianQuadrature:
    r"""A Bayesian quadrature model to solve integrals of the type.

    .. math:: F = \int_a^b f(x) d \mu(x),

    Parameters
    ----------
    kernel:
        the kernel used for the GP model
    policy:
        the policy for acquiring nodes for function evaluations
    """

    def __init__(self, kernel: Kernel, policy: Callable) -> None:
        self.kernel = kernel
        self.policy = policy

    @classmethod
    def instantiate_default(
        cls,
        input_dim: int,
        kernel: Optional[Kernel] = None,
        method: str = "vanilla",
        policy: str = "bmc",
    ) -> "BayesianQuadrature":
        if method != "vanilla":
            raise NotImplementedError
        if kernel is None:
            kernel = ExpQuad(input_dim=input_dim)
        if policy == "bmc":
            policy = sample_from_measure
        else:
            raise NotImplementedError(
                "Policies outside random sampling are not available at the moment"
            )
        return cls(kernel=kernel, policy=policy)

    def integrate(
        self, fun: Callable, measure: IntegrationMeasure, nevals: int
    ) -> Tuple[Normal, Dict]:
        r"""Integrate the function ``fun``.

        Parameters
        ----------
        fun:
            integrand function :math:`f`
        measure :
            integration measure :math:`\mu`
        nevals :
            Number of function evaluations.

        Returns
        -------
        F : RandomVariable
            The integral of ``fun`` from ``a`` to ``b``.
        info : dict
            Information on the performance of the method.
        """

        # Acquisition policy
        nodes = self.policy(nevals, measure)
        y = fun(nodes)

        # compute integral mean and variance
        # Define kernel embedding
        kernel_embedding = get_kernel_embedding(self.kernel, measure)
        gram = self.kernel(nodes, nodes)
        kernel_mean = kernel_embedding.kernel_mean(nodes)
        initial_error = kernel_embedding.kernel_variance()

        weights = self._solve_gram(gram, kernel_mean)

        integral_mean = np.squeeze(weights.T @ y)
        integral_variance = initial_error - weights.T @ kernel_mean

        F = Normal(integral_mean, integral_variance)

        # Information on result
        info = {"model_fit_diagnostic": None}

        return F, info

    # The following functions are here for the minimal version
    # and shall be factored out once BQ is expanded.
    # 1. acquisition policy
    # 2. GP inference

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


def sample_from_measure(nevals: int, measure: IntegrationMeasure) -> np.ndarray:
    r"""Acquisition policy: random samples from the integration measure

    Parameters
    ----------
    nevals : int
        Number of function evaluations.

    measure :
            integration measure :math:`\mu`

    Returns
    -------
    x : np.ndarray
        nodes where the integrand will be evaluated
    """
    return measure.sample(nevals).reshape(nevals, -1)
