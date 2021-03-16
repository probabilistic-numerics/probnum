"""Probabilistic numerical methods for solving integrals."""

from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from probnum.kernels import Kernel
from probnum.random_variables import Normal

from .._integration_measures import IntegrationMeasure
from .._kernel_embeddings import get_kernel_embedding


class BayesianQuadrature:
    """A Bayesian quadrature model to solve integrals of the type.

    .. math:: \\int f(x) p(x) dx

    This class is designed to be subclassed by implementations of Bayesian quadrature
    with an :meth:`integrate` method.

    Parameters
    ----------
    fun: Callable
        integrand function :math:`f`
    kernel: Kernel
        the kernel used for the GP model
    measure : IntegrationMeasure
        integration measure :math:`p`
    """

    def __init__(self, fun: Callable, kernel: Kernel, measure: IntegrationMeasure):
        self.fun = fun
        self.kernel = kernel
        self.measure = measure

        # Define kernel embedding
        self.kernel_embedding = get_kernel_embedding(
            kernel=self.kernel, measure=self.measure
        )

    def integrate(self, nevals: int):
        """Integrate the function ``fun``.

        Parameters
        ----------
        nevals : int
            Number of function evaluations.

        Returns
        -------
        F : RandomVariable
        The integral of ``func`` from ``a`` to ``b``.
        fun0 : RandomProcess
            Stochastic process modelling the function to be integrated after ``neval``
            observations.
        info : dict
            Information on the performance of the method.
        """

        # Acquisition policy
        nodes = self._policy(nevals)
        y = self.fun(nodes)

        # compute integral mean and variance
        gram = self.kernel(nodes, nodes)
        kernel_mean = self.kernel_embedding.kernel_mean(nodes)
        initial_error = self.kernel_embedding.kernel_variance()

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

    def _policy(self, nevals: int):
        """Acquisition policy for obtaining locations where to evaluate the function.

        Parameters
        ----------
        nevals : int
            Number of function evaluations.

        Returns
        -------
        x : np.ndarray
            nodes where the integrand will be evaluated
        """
        return self.measure.sample(nevals).reshape(nevals, -1)

    @staticmethod
    def _solve_gram(gram: np.ndarray, rhs: np.ndarray):
        """Solve the linear system of the form.

        .. math:: Kx=b,

        Parameters
        ----------
        gram : np.ndarray
            symmetric pos. def. kernel Gram matrix :math:`K`, shape (nevals, nevals)
        rhs : np.ndarray
            right-hand-side :math:`b`, matrix or vector, shape (nevals, ...)

        Returns
        -------
        x:  np.ndarray
            The solution to the linear system :math:`K x = b`
        """
        jitter = 1.0e-6
        chol_gram = cho_factor(gram + jitter * np.eye(gram.shape[0]))
        return cho_solve(chol_gram, rhs)
