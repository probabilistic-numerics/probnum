"""
Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian quadrature,
meaning a model over the integrand is constructed in order to actively select evaluation
points of the integrand to estimate the value of the integral. Bayesian quadrature
methods return a random variable with a distribution, specifying the belief about the
true value of the integral.
"""

from probnum.quad.quadrature import Quadrature


def bayesquad(fun, fun0, bounds, nevals=None, type="vanilla", **kwargs):
    """
    One-dimensional Bayesian quadrature.

    Parameters
    ----------
    fun : function
        Function to be integrated.
    fun0 : RandomProcess
        Stochastic process modelling the function to be integrated.
    bounds : ndarray, shape=(2,)
        Lower and upper limit of integration.
    nevals : int
        Number of function evaluations.
    type : str
        Type of Bayesian quadrature to use. The available options are

        ====================  ===========
         vanilla              ``vanilla``
         WSABI                ``wsabi``
        ====================  ===========

    kwargs : optional
        Keyword arguments passed on.

    Returns
    -------
    F : RandomVariable
        The integral of ``func`` from ``a`` to ``b``.
    fun0 : RandomProcess
        Stochastic process modelling the function to be integrated after ``neval``
        observations.
    info : dict
        Information on the performance of the method.

    References
    ----------

    See Also
    --------
    nbayesquad : N-dimensional Bayesian quadrature.
    """

    # Choose Method
    bqmethod = None
    if type == "vanilla":
        bqmethod = VanillaBayesianQuadrature()
    elif type == "wsabi":
        bqmethod = WSABIBayesianQuadrature()

    # Integrate
    F, fun0, info = bqmethod.integrate(
        fun=fun, fun0=fun0, nevals=nevals, domain=bounds, **kwargs
    )

    return F, fun0, info


def nbayesquad(fun, fun0, domain, nevals=None, type=None, **kwargs):
    """
    N-dimensional Bayesian quadrature.

    Parameters
    ----------
    fun : function
        Function to be integrated.
    fun0 : RandomProcess
        Stochastic process modelling the function to be integrated.
    domain : ndarray
        Domain of integration.
    nevals : int
        Number of function evaluations.
    type : str
        Type of Bayesian quadrature to use. The available options are

        ====================  ===========
         vanilla              ``vanilla``
         WASABI               ``wasabi``
        ====================  ===========

    kwargs : optional
        Keyword arguments passed on.

    Returns
    -------
    F : RandomVariable
        The integral of ``func`` on the domain.
    fun0 : RandomProcess
        Stochastic process modelling the function to be integrated after ``neval``
        observations.
    info : dict
        Information on the performance of the method.

    References
    ----------

    See Also
    --------
    bayesquad : 1-dimensional Bayesian quadrature.
    """
    raise NotImplementedError


class BayesianQuadrature(Quadrature):
    """
    An abstract base class for Bayesian quadrature methods.

    This class is designed to be subclassed by implementations of Bayesian quadrature
    with an :meth:`integrate` method.
    """

    def integrate(self, fun, fun0, domain, nevals, **kwargs):
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
        kwargs

        Returns
        -------

        """
        raise NotImplementedError


class VanillaBayesianQuadrature(BayesianQuadrature):
    """
    Vanilla Bayesian quadrature in 1D.
    """

    def integrate(self, fun, fun0, domain, nevals, **kwargs):
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

    def integrate(self, fun, fun0, domain, nevals, **kwargs):
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

        raise NotImplementedError
