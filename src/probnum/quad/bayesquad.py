"""
Bayesian Quadrature.

"""

import abc

__all__ = ["bayesquad"]


def bayesquad(func, func0, a, b, nevals=None, type="vanilla", **kwargs):
    """
    One dimensional Bayesian Quadrature.

    Parameters
    ----------
    func : function
        Symbolic representation of the function to be integrated.
    func0 : RandomProcess
        Stochastic process modelling function to be integrated.
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
        ====================  =========

    kwargs : optional
        Keyword arguments passed on.

    Returns
    -------
    F : RandomVariable
        The integral of ``fun`` from ``a`` to ``b``.
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
        bqmethod = _WASABIBayesian_Quadrature(func=func, func0=func0)

    # Integrate
    F, func0, info = bqmethod.integrate(nevals=nevals, **kwargs)

    return F, func0, info


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

    def integrate(self, **kwargs):
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
        self.func = func
        self.func0 = func0

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


class _WASABIBayesian_Quadrature(_BayesianQuadrature):
    """
    WASABI
    """

    def __init__(self, func, func0):
        self.func = func
        self.func0 = func0

    def integrate(self, nevals, **kwargs):
        raise NotImplementedError
