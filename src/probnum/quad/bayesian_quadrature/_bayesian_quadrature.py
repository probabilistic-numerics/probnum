"""Bayesian Quadrature."""

import abc


class BayesianQuadrature(abc.ABC):
    """
    An abstract base class for Bayesian quadrature methods.

    This class is designed to be subclassed by implementations of Bayesian quadrature
    with an :meth:`integrate` method.

    Parameters
    ----------
    fun0
        Stochastic process modelling function to be integrated.
    """

    def __init__(self, fun0):
        self.fun0 = fun0

    def integrate(self, fun, domain, nevals, **kwargs):
        """
        Integrate the function ``fun``.

        Parameters
        ----------
        fun : function
            Function to be integrated.
        domain : ndarray, shape=()
            Domain to integrate over.
        nevals : int
            Number of function evaluations.
        kwargs

        Returns
        -------

        """
        raise NotImplementedError


class VanillaBayesianQuadrature(BayesianQuadrature):
    """
    Vanilla Bayesian quadrature in 1D.
    """

    def integrate(self, fun, domain, nevals, **kwargs):
        """
        Integrate the function ``fun``.

        Parameters
        ----------
        fun : function
            Function to be integrated.
        fun0 : RandomProcess
            Stochastic process modelling function to be integrated.
        domain : ndarray, shape=()
            Domain to integrate over.
        nevals : int
            Number of function evaluations.

        Returns
        -------

        """

        # Initialization
        F = None

        # Iteration
        for _ in range(nevals):
            # Predictive Distribution
            # fun_pred = None

            # Observation

            # Update Distribution
            self.fun0 = None

        # Information on result
        info = {"model_fit_diagnostic": None}

        return F, self.fun0, info


class WSABIBayesianQuadrature(BayesianQuadrature):
    """
    Warped Sequential Active Bayesian Integration (WSABI).
    """

    def integrate(self, fun, domain, nevals, **kwargs):
        """
        Integrate the function ``fun``.

        Parameters
        ----------
        fun : function
            Function to be integrated.
        domain : ndarray, shape=()
            Domain to integrate over.
        nevals : int
            Number of function evaluations.

        Returns
        -------

        """

        raise NotImplementedError
