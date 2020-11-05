"""
Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian quadrature,
meaning a model over the integrand is constructed in order to actively select evaluation
points of the integrand to estimate the value of the integral. Bayesian quadrature
methods return a random variable with a distribution, specifying the belief about the
true value of the integral.
"""

from .bq_methods._bayesian_quadrature import (
    VanillaBayesianQuadrature,
    WSABIBayesianQuadrature,
)


def bayesquad(fun, fun0, domain, nevals=None, method="vanilla", **kwargs):
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
    method : str
        Type of Bayesian quadrature to use. The available options are

        ====================  ===========
         vanilla              ``vanilla``
         WASABI               ``wasabi``
        ====================  ===========

    kwargs : optional
        Keyword arguments passed on.

    Returns
    -------
    integral : RandomVariable
        The integral of ``func`` on the domain.
    fun0 : RandomProcess
        Stochastic process modelling the function to be integrated after ``neval``
        observations.
    info : dict
        Information on the performance of the method.

    References
    ----------
    """

    # Choose Method
    bqmethod = None
    if method == "vanilla":
        bqmethod = VanillaBayesianQuadrature(fun0=fun0)
    elif method == "wsabi":
        bqmethod = WSABIBayesianQuadrature(fun0=fun0)

    # Integrate
    integral, fun0, info = bqmethod.integrate(
        fun=fun, nevals=nevals, domain=domain, **kwargs
    )

    return integral, fun0, info
