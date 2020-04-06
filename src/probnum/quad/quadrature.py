"""
Quadrature, i.e. numerical integration.

This module provides an abstract base class defining quadrature methods.
"""

import abc


def quad(func, bounds, type=None):
    """
    One-dimensional numerical integration.

    Parameters
    ----------
    func : function
        Function to be integrated.
    bounds : ndarray, shape=(2,)
        Domain of integration.
    type : str
        Type of quadrature to use. The available options are

        ====================  ===========
         Bayesian              ``bayes``
         Clenshaw-Curtis       ``cc``
        ====================  ===========

    Returns
    -------
    F : RandomVariable
        The integral of ``func`` within the given bounds.
    """
    raise NotImplementedError


def nquad(func, domain, type=None):
    """
    N-dimensional numerical integration.

    Parameters
    ----------
    func : function
        Function to be integrated.
    domain : ndarray, shape=(d,2)
        Domain of integration.
    type : str
        Type of quadrature to use. The available options are

        ====================  ===========
         Bayesian              ``bayes``
         Clenshaw-Curtis       ``cc``
        ====================  ===========

    Returns
    -------
    F : RandomVariable
        The integral of ``func`` on the domain.
    """
    raise NotImplementedError


class Quadrature(abc.ABC):
    """
    An abstract base class for quadrature methods.

    This class is designed to be subclassed by quadrature implementations.
    """

    def __init__(self):
        """
        """

    def integrate(self, func, **kwargs):
        """
        Numerically integrate the given function.

        Parameters
        ----------
        func : function
            Function to be integrated.
        kwargs

        Returns
        -------

        """
        raise NotImplementedError
