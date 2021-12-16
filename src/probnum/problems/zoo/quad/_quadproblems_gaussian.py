import numpy as np
from scipy import special
from scipy.stats import norm

from probnum.typing import FloatArgType

__all__ = [
    "genz_continuous_gaussian",
    "sum_polynomials",
]


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

    if isinstance(var, float) is False or var <= 0.0:
        raise TypeError(f"The variance should be a positive float.")

    def newfunc(x):
        return func(norm.cdf((x - mean) / var))

    return newfunc


def genz_continuous_gaussian(
    dim: int,
    a: np.ndarray = None,
    u: np.ndarray = None,
    mean: FloatArgType = 0.0,
    var: FloatArgType = 1.0,
) -> QuadratureProblem:
    """This integrand is a transformation of the Genz 'continuous' test function on.

    [0,1]^d; i.e.

    .. math::'h(x)=f(\Phi((x-mean)/var))'
    where :math:'\Phi(x)' an elementwise application of the Gaussian cummulative distribution function and
    .. math::  f(x) = \exp(- \sum_{i=1}^d a_i |x_i - u_i|).

    The integrand is integrated against a N(mean, var * I).


    Parameters
    ----------
        dim
            Dimension of the domain
        a
            First set of parameters affecting the difficulty of the integration problem.
        u
            Second set of parameters affecting the difficulty of the integration problem.
            All entries should be in [0,1].
    References
    ----------
        https://www.sfu.ca/~ssurjano/cont.html
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(5.0, dim)

    if u is None:
        u = np.broadcast_to(0.5, dim)

    # Check that the parameters have valid values and shape
    if a.shape != (dim,):
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {(dim,)}."
        )

    if u.shape != (dim,):
        raise ValueError(
            f"Invalid shape {u.shape} for parameter `u`. Expected {(dim,)}."
        )

    if np.any(u < 0.0) or np.any(u > 1):
        raise ValueError(f"The parameters `u` must lie in the interval [0.0, 1.0].")

    def integrand_uniform(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        nonlocal a, u
        n = x.shape[0]

        # Check that the input points have valid values and shape
        if x.shape != (n, dim):
            raise ValueError(
                f"Invalid shape {x.shape} for input points `x`. Expected (n, {dim})."
            )

        if np.any(x < 0.0) or np.any(x > 1):
            raise ValueError(f"The input points `x` must lie in the box [0.0, 1.0]^d.")

        # Reshape u into an (n,dim) array with identical rows
        u = np.repeat(u.reshape([1, dim]), n, axis=0)

        # Compute function values
        f = np.exp(-np.sum(a * np.abs(x - u), axis=1))

        return f.reshape((n, 1))

    integrand = uniform_to_gaussian(func=integrand_uniform, mean=mean, var=var)

    solution = np.prod((2.0 - np.exp(-a * u) - np.exp(a * (u - 1))) / a)

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(-np.Inf, dim),
        upper_bd=np.broadcast_to(np.Inf, dim),
        output_dim=None,
        solution=solution,
    )


def sum_polynomials(
    dim: int, a: np.ndarray = None, b: np.ndarray = None, var: float = 1.0
) -> QuadratureProblem:
    """Integrand taking the form of a sum of polynomials over :math:'\mathbb{R}^d'.

    .. math::  f(x) = \sum_{j=0}^p \prod_{i=1}^dim a_{ji} x_i^{b_ji}

    The integrand is integrated against a multivariate normal N(0,var * I_d).

    Parameters
    ----------
        dim
            Dimension d of the integration domain
        a
            2d-array of size (p+1)xd giving coefficients of the polynomials.
        b
            2d-array of size (p+1)xd giving orders of the polynomials. All entries should be natural numbers.
        var
            diagonal elements of the covariance function.

    Returns
    -------
        f
            array of function evaluations at points 'x'.


    References
    ----------
        Si, S., Oates, C. J., Duncan, A. B., Carin, L. & Briol. F-X. (2021). Scalable control variates for Monte Carlo methods via stochastic optimization.
        Proceedings of the 14th Monte Carlo and Quasi-Monte Carlo Methods (MCQMC) conference 2020. arXiv:2006.07487.
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(1.0, (1, dim))
    if b is None:
        b = np.broadcast_to(1, (1, dim))

    # Check that the parameters have valid values and shape
    if a.shape[1] != dim:
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {dim} columns."
        )

    if b.shape[1] != dim:
        raise ValueError(
            f"Invalid shape {b.shape} for parameter `b`. Expected {dim} columns."
        )

    if np.any(b < 0):
        raise ValueError(f"The parameters `b` must be non-negative.")

    def integrand(x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x
            Array of points at which to evaluate the integrand of size (n,dim).
            All entries should be in [0,1].
        Returns
        -------
        f
            array of integrand evaluations at points in 'x'.
        """
        n = x.shape[0]

        # Compute function values
        f = np.sum(
            np.prod(
                a[np.newaxis, :, :] * (x[:, np.newaxis, :] ** b[np.newaxis, :, :]),
                axis=2,
            ),
            axis=1,
        )

        # Return function values as a 2d array with one column.
        return f.reshape((n, 1))

    # Return function values as a 2d array with one column.
    delta = (np.remainder(b, 2) - 1) ** 2
    doublefact = special.factorial2(b - 1)
    solution = np.sum(np.prod(a * delta * (var ** b) * doublefact, axis=1))

    return QuadratureProblem(
        integrand=integrand,
        lower_bd=np.broadcast_to(-np.Inf, dim),
        upper_bd=np.broadcast_to(np.Inf, dim),
        output_dim=None,
        solution=solution,
    )
