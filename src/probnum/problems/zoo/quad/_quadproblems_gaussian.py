import numpy as np
from scipy import special

from probnum.typing import FloatArgType


def sum_polynomials(
    x: np.ndarray, a: np.ndarray = None, b: np.ndarray = None
) -> np.ndarray:
    """Integrand taking the form of a sum of polynomials over :math:'\mathbb{R}^d'.

    .. math::  f(x) = \sum_{j=0}^p \prod_{i=1}^dim a_{ji} x_i^{b_ji}

    Parameters
    ----------
        x
            Array of points at which to evaluate the function.
            Each row is a point and column corresponds to dimensions.
        a
            2d-array of size (p+1)xd giving coefficients of the polynomials.
        b
            2d-array of size (p+1)xd giving orders of the polynomials. All entries should be natural numbers.

    Returns
    -------
        f
            array of function evaluations at points 'x'.


    References
    ----------
        Si, S., Oates, C. J., Duncan, A. B., Carin, L. & Briol. F-X. (2021). Scalable control variates for Monte Carlo methods via stochastic optimization.
        Proceedings of the 14th Monte Carlo and Quasi-Monte Carlo Methods (MCQMC) conference 2020. arXiv:2006.07487.
    """

    (n, dim) = x.shape

    # Specify default values of parameters a and u
    if a is None:
        a = np.ones((1, dim))
    if u is None:
        b = np.ones((1, dim), dtype=int)

    # Check that the parameters have valid values and shape
    assert len(a.shape) == 2 and a.shape[1] == dim
    assert len(b.shape) == 2 and b.shape[1] == dim
    assert np.all(b > -1)

    # Compute function values
    f = np.sum(
        np.prod(
            a[np.newaxis, :, :] * (x[:, np.newaxis, :] ** b[np.newaxis, :, :]), axis=2
        ),
        axis=1,
    )

    # Return function values as a 2d array with one column.
    return f.reshape((n, 1))


def integral_sum_polynomials(
    a: np.ndarray, b: np.ndarray, var: FloatArgType
) -> FloatArgType:
    """
    Integral of the sum of polynomials against a multivariate Gaussian N(0,var * I_d)

    Parameters
    ----------
        a
            2d-array of size (p+1)xd giving coefficients of the polynomials.
        b
            2d-array of size (p+1)xd giving orders of the polynomials. All entries should be natural numbers.
        var
            diagonal element for the covariance matrix of the multivariate Gaussian

    Returns
    -------
        Value of the integral


    References
    ----------
        Si, S., Oates, C. J., Duncan, A. B., Carin, L. & Briol. F-X. (2021). Scalable control variates for Monte Carlo methods via stochastic optimization.
        Proceedings of the 14th Monte Carlo and Quasi-Monte Carlo Methods (MCQMC) conference 2020. arXiv:2006.07487.
    """

    # Check that the parameters have valid values and shape
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert np.all(b > -1)
    assert isinstance(var, float) and var >= 0.0

    delta = (np.remainder(b, 2) - 1) ** 2
    doublefact = special.factorial2(b - 1)

    return np.sum(np.prod(a * delta * (var ** b) * doublefact, axis=1))
