"""
Bayesian Quadrature.

This module provides routines to integrate functions through Bayesian quadrature, meaning a model over the integrand
is constructed in order to actively select evaluation points of the integrand to estimate the value of the integral.
Bayesian quadrature methods return a random variable with a distribution, specifying the belief about the true value of
the integral.
"""

import abc
import numpy as np

__all__ = ["bayesquad", "nbayesquad"]


def bayesquad(func, func0, a, b, nevals=None, type="vanilla", **kwargs):
    """
    One dimensional Bayesian Quadrature.

    Parameters
    ----------
    func : function
        Function to be integrated.
    func0 : RandomProcess
        Stochastic process modelling the function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
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
        bqmethod = _VanillaBayesianQuadrature(func=func, func0=func0)
    elif type == "wasabi":
        bqmethod = _WASABIBayesianQuadrature(func=func, func0=func0)

    # Integrate
    F, func0, info = bqmethod.integrate(nevals=nevals, domain=np.array([a, b]), **kwargs)

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
    domain :
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


class _BayesianQuadrature(abc.ABC):
    """
    An abstract base class for Bayesian Quadrature methods.
    """

    def __init__(self, func, func0):
        """
        Parameters
        ----------
        func : function
            Symbolic representation of the function to be integrated.
        func0 : RandomProcess
            Stochastic process modelling function to be integrated.
        """
        self.func = func
        self.func0 = func0

    def integrate(self, domain, **kwargs):
        """
        Integrate the function ``func``.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        raise NotImplementedError


class _VanillaBayesianQuadrature(_BayesianQuadrature):
    """
    Vanilla Bayesian Quadrature in 1D.
    """

    def __init__(self, func, func0):
        super().__init__(func=func, func0=func0)

    def integrate(self, nevals, **kwargs):
        """

        Parameters
        ----------
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


class _WASABIBayesianQuadrature(_BayesianQuadrature):
    """
    Weighted Adaptive Surrogate Approximations for Bayesian Inference (WASABI).
    """

    def __init__(self, func, func0):
        super().__init__(func=func, func0=func0)

    def integrate(self, nevals, **kwargs):
        raise NotImplementedError
