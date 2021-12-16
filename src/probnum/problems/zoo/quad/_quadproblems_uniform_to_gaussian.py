from typing import Callable

import numpy as np
from scipy.stats import norm

from probnum.typing import FloatArgType, IntArgType

__all__ = ["uniform_to_gaussian"]


def uniform_to_gaussian(
    func: Callable[[np.ndarray], np.ndarray],
    mean: FloatArgType = 0.0,
    var: FloatArgType = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Transforming an integrand suitable for integration against Lebesgue measure on.

    [0,1]^d to an integrand suitable for integration against a d-dimensional Gaussian of
    the form N(mean*(1,...,1),var^2 I_d).

    Using the change of variable formula, we have that

    .. math::  \int_{[0,1]^d} f(x) dx = \int_{\mathbb{R}^d} h(x) \phi(x) dx

    where :math:'h(x)=f(\Phi((x-mean)/var))', :math:'\phi(x)' is the Gaussian probability density function
    and :math:'\Phi(x)' an elementwise application of the Gaussian cummulative distribution function.

    This function therefore takes f as input and returns h.

    Parameters
    ----------
        func
            A test function which takes inputs in [0,1]^d and returns an array of function values.
        mean
            Mean of the Gaussian distribution.
        var
            Diagonal element for the covariance matrix of the Gaussian distribution.

    Returns
    -------
        newfunc
            A transformed test function taking inputs in :math:'\mathbb{R}^d'.

    Examples
    --------
    >>> Bratley1992_gaussian = uniform_to_gaussian(Bratley1992)


    References
    ----------
        Si, S., Oates, C. J., Duncan, A. B., Carin, L. & Briol. F-X. (2021). Scalable control variates for Monte Carlo methods via stochastic optimization.
        Proceedings of the 14th Monte Carlo and Quasi-Monte Carlo Methods (MCQMC) conference 2020. arXiv:2006.07487.
    """

    # mean and var should be either one-dimensional, or an array of dimension d
    if isinstance(mean, float) is False:
        raise TypeError(f"The mean parameter should be a float.")

    if iisinstance(var, float) is False or var <= 0.0:
        raise TypeError(f"The variance should be a positive float.")

    def newfunc(x):
        return func(norm.cdf((x - mean) / var))

    return newfunc
