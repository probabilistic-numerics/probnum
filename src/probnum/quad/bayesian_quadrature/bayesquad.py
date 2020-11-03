"""
Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian quadrature,
meaning a model over the integrand is constructed in order to actively select evaluation
points of the integrand to estimate the value of the integral. Bayesian quadrature
methods return a random variable with a distribution, specifying the belief about the
true value of the integral.
"""

from ._bayesian_quadrature import VanillaBayesianQuadrature, WSABIBayesianQuadrature


def bayesquad(fun, fun0, domain, nevals=None, type="vanilla", **kwargs):
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
    """

    # Choose Method
    bqmethod = None
    if type == "vanilla":
        bqmethod = VanillaBayesianQuadrature()
    elif type == "wsabi":
        bqmethod = WSABIBayesianQuadrature()

    # Integrate
    F, fun0, info = bqmethod.integrate(
        fun=fun, fun0=fun0, nevals=nevals, domain=domain, **kwargs
    )

    return F, fun0, info
