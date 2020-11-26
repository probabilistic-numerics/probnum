"""Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian
quadrature, meaning a model over the integrand is constructed in order
to actively select evaluation points of the integrand to estimate the
value of the integral. Bayesian quadrature methods return a random
variable, specifying the belief about the true value of the integral.
"""

from .bq_methods import BayesianQuadrature, WarpedBayesianQuadrature


def bayesquad(fun, fun0, domain, measure, nevals=None, method="vanilla"):
    """N-dimensional Bayesian quadrature.

    Parameters
    ----------
    fun :
        Function to be integrated.
    fun0 :
        Stochastic process modelling the function to be integrated.
    domain :
        Domain of integration.
    measure :
        Measure to integrate against.
    nevals :
        Number of function evaluations.
    method :
        Type of Bayesian quadrature to use. The available options are

        ====================  ===========
         vanilla              ``vanilla``
         WSABI                ``wsabi``
        ====================  ===========

    Returns
    -------
    integral :
        The integral of ``func`` on the domain.
    fun0 :
        Stochastic process modelling the function to be integrated after ``neval``
        observations.
    info :
        Information on the performance of the method.

    References
    ----------
    """

    # Choose Method
    bqmethod = None
    if method == "vanilla":
        bqmethod = BayesianQuadrature(fun0=fun0)
    elif method == "wsabi":
        bqmethod = WarpedBayesianQuadrature(fun0=fun0)

    # Integrate
    integral, fun0, info = bqmethod.integrate(
        fun=fun, nevals=nevals, domain=domain, measure=measure
    )

    return integral, fun0, info
