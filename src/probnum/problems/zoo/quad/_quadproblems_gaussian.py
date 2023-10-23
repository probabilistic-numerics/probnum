"""Test problems for integration against a Gaussian measure."""

from typing import Callable, Union

import numpy as np
from scipy import special
from scipy.stats import norm

from probnum.problems import QuadratureProblem
from probnum.quad.integration_measures import GaussianMeasure
from probnum.typing import FloatLike

__all__ = [
    "uniform_to_gaussian_quadprob",
    "sum_polynomials",
]


# Construct transformation of the integrand
def uniform_to_gaussian_integrand(
    fun: Callable[[np.ndarray], np.ndarray],
    mean: Union[float, np.floating, np.ndarray] = 0.0,
    std: Union[float, np.floating, np.ndarray] = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    # mean and var should be either one-dimensional
    if isinstance(mean, np.ndarray):
        if len(mean.shape) != 1:
            raise TypeError(
                "The mean parameter should be a float or a d-dimensional array."
            )

    if isinstance(std, np.ndarray):
        if len(std.shape) != 1:
            raise TypeError(
                "The std parameter should be a float or a d-dimensional array."
            )

    def new_func(x):
        return fun(norm.cdf(x, loc=mean, scale=std))

    return new_func


def uniform_to_gaussian_quadprob(
    quadprob: QuadratureProblem,
    mean: Union[float, np.floating, np.ndarray] = 0.0,
    std: Union[float, np.floating, np.ndarray] = 1.0,
) -> QuadratureProblem:
    r"""Creates a new QuadratureProblem for integration against a Gaussian on
    :math:`\mathbb{R}^d` by using an existing QuadratureProblem whose integrand is
    suitable for integration against the Lebesgue measure on :math:`[0,1]^d`.

    The multivariate Gaussian is of the form :math:`\mathcal{N}(mean \cdot (1, \dotsc,
    1)^\top, var^2 \cdot I_d)`.

    Using the change of variable formula, we have that

    .. math::  \int_{[0,1]^d} f(x) dx = \int_{\mathbb{R}^d} h(x) \phi(x) dx

    where :math:`h(x)=f(\Phi((x-mean)/var))`, :math:`\phi(x)` is the Gaussian
    probability density function and :math:`\Phi(x)` an elementwise application of the
    Gaussian cumulative distribution function. See [1]_.

    Parameters
    ----------
    quadprob
        A QuadratureProblem instance which includes an integrand defined on [0,1]^d
    mean
        Mean of the Gaussian distribution. If `float`, mean is set to the same value
        across all dimensions. Else, specifies the mean as a d-dimensional array.
    std
        Diagonal element for the covariance matrix of the Gaussian distribution. If
        `float`, the covariance matrix has the same diagonal value for all dimensions.
        Else, specifies the covariance matrix via a d-dimensional array.

    Returns
    -------
    problem
        A new Quadrature Problem instance with a transformed integrand taking inputs in
        :math:`\mathbb{R}^d`.

    Raises
    ------
    ValueError
        If the original quadrature problem is over a domain other than [0, 1]^d or if it
        does not have a scalar solution.

    Example
    -------
    Convert the uniform continuous Genz problem to a Gaussian quadrature problem.

    >>> import numpy as np
    >>> from probnum.problems.zoo.quad import genz_continuous
    >>> gaussian_quadprob = uniform_to_gaussian_quadprob(genz_continuous(1))
    >>> gaussian_quadprob.fun(np.array([[0.]]))
    array([[1.]])

    References
    ----------
    .. [1] Si, S., Oates, C. J., Duncan, A. B., Carin, L. & Briol. F-X. (2021). Scalable
       control variates for Monte Carlo methods via stochastic optimization. Proceedings
       of the 14th Monte Carlo and Quasi-Monte Carlo Methods (MCQMC) conference 2020.
       arXiv:2006.07487.
    """
    lower_bd, upper_bd = quadprob.measure.domain

    # Check that the original quadrature problem was defined on [0,1]^d
    if np.any(lower_bd != 0.0):
        raise ValueError("quadprob is not an integration problem over [0,1]^d")
    if np.any(upper_bd != 1.0):
        raise ValueError("quadprob is not an integration problem over [0,1]^d")

    # Check that the original quadrature problem has a scalar valued solution
    if np.ndim(quadprob.solution) != 0:
        raise ValueError("The solution of quadprob is not a scalar.")

    dim = lower_bd.shape[0]

    cov = std**2
    if isinstance(std, np.ndarray):
        cov = np.eye(dim) * cov
    gaussian_measure = GaussianMeasure(mean=mean, cov=cov, input_dim=dim)
    return QuadratureProblem(
        fun=uniform_to_gaussian_integrand(fun=quadprob.fun, mean=mean, std=std),
        measure=gaussian_measure,
        solution=quadprob.solution,
    )


def sum_polynomials(
    dim: int, a: np.ndarray = None, b: np.ndarray = None, var: FloatLike = 1.0
) -> QuadratureProblem:
    r"""Quadrature problem with an integrand taking the form of a sum of polynomials
    over :math:`\mathbb{R}^d`.

    .. math::  f(x) = \sum_{j=0}^p \prod_{i=1}^dim a_{ji} x_i^{b_ji}

    The integrand is integrated against a multivariate normal :math:`\mathcal{N}(0,var *
    I_d)`. See [1]_.

    Parameters
    ----------
    dim
        Dimension d of the integration domain
    a
        2d-array of size (p+1)xd giving coefficients of the polynomials.
    b
        2d-array of size (p+1)xd giving orders of the polynomials. All entries
        should be natural numbers.
    var
        diagonal elements of the covariance function.

    Returns
    -------
    f
        array of size (n,1) giving integrand evaluations at points in 'x'.

    Raises
    ------
    ValueError
        If the given parameters have the wrong shape or contain invalid values.

    References
    ----------
    .. [1] Si, S., Oates, C. J., Duncan, A. B., Carin, L. & Briol. F-X. (2021). Scalable
       control variates for Monte Carlo methods via stochastic optimization. Proceedings
       of the 14th Monte Carlo and Quasi-Monte Carlo Methods (MCQMC) conference 2020.
       arXiv:2006.07487.
    """

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(1.0, (1, dim))
    if b is None:
        b = np.broadcast_to(1, (1, dim))

    if len(a.shape) != 2:
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. "
            f"Expected parameters of shape (p+1)xdim"
        )

    if len(b.shape) != 2:
        raise ValueError(
            f"Invalid shape {b.shape} for parameter `b`. "
            f"Expected parameters of shape (p+1)xdim"
        )

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
        raise ValueError("The parameters `b` must be non-negative.")

    def integrand(x: np.ndarray) -> np.ndarray:
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
    solution = np.sum(np.prod(a * delta * (var**b) * doublefact, axis=1))
    if isinstance(var, float):
        mean = 0.0
    else:
        mean = np.zeros(dim)

    gaussian_measure = GaussianMeasure(mean=mean, cov=var, input_dim=dim)
    return QuadratureProblem(
        fun=integrand,
        measure=gaussian_measure,
        solution=solution,
    )
