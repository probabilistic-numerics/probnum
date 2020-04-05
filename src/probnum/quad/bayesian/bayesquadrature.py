"""
Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian quadrature, meaning a model over the integrand
is constructed in order to actively select evaluation points of the integrand to estimate the value of the integral.
Bayesian quadrature methods return a random variable with a distribution, specifying the belief about the true value of
the integral.
"""

import numpy as np
from probnum.quad.quadrature import Quadrature


def bayesquad(func, func0, bounds, nevals=None, type="vanilla", **kwargs):
    """
    One dimensional Bayesian Quadrature.

    Parameters
    ----------
    func : function
        Function to be integrated.
    func0 : RandomProcess
        Stochastic process modelling the function to be integrated.
    bounds : ndarray, shape=(2,)
        Lower and upper limit of integration.
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
        The integral of ``func`` from ``a`` to ``b``.
    func0 : RandomProcess
        Stochastic process modelling the function to be integrated after ``neval`` observations.
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
    elif type == "wasabi":
        bqmethod = WASABIBayesianQuadrature()

    # Integrate
    F, func0, info = bqmethod.integrate(func=func, func0=func0, nevals=nevals, domain=bounds, **kwargs)

    return F, func0, info


def nbayesquad(func, func0, domain, nevals=None, type=None, **kwargs):
    """
    N-dimensional Bayesian Quadrature.

    Parameters
    ----------
    func : function
        Function to be integrated.
    func0 : RandomProcess
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
    func0 : RandomProcess
        Stochastic process modelling the function to be integrated after ``neval`` observations.
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
    An abstract base class for Bayesian Quadrature methods.

    This class is designed to be subclassed by implementations of Bayesian quadrature with an :meth:`integrate` method.
    """

    def __init__(self):
        super().__init__()

    def integrate(self, func, func0, domain, nevals, **kwargs):
        """
        Integrate the function ``func``.

        Parameters
        ----------
        func : function
            Function to be integrated.
        func0 : RandomProcess
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
    Vanilla Bayesian Quadrature in 1D.
    """

    def __init__(self):
        super().__init__()

    def integrate(self, func, func0, domain, nevals, **kwargs):
        """
        Integrate the function ``func``.

        Parameters
        ----------
        func : function
            Function to be integrated.
        func0 : RandomProcess
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
        for i in range(nevals):
            # Predictive Distribution
            func_pred = None

            # Observation

            # Update Distribution
            self.func0 = None

        # Information on result
        info = {
            "model_fit_diagnostic": None
        }

        return F, self.func0, info


class WASABIBayesianQuadrature(BayesianQuadrature):
    """
    Weighted Adaptive Surrogate Approximations for Bayesian Inference (WASABI).
    """

    def __init__(self):
        super().__init__()

    def integrate(self, func, func0, domain, nevals, **kwargs):
        """
        Integrate the function ``func``.

        Parameters
        ----------
        func : function
            Function to be integrated.
        func0 : RandomProcess
            Stochastic process modelling function to be integrated.
        domain : ndarray, shape=()
            Domain to integrate over.
        nevals : int
            Number of function evaluations.

        Returns
        -------

        """

        raise NotImplementedError
