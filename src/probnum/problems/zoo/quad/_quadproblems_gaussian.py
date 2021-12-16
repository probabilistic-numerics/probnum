import numpy as np
from scipy import special
from scipy.stats import norm

from probnum.typing import FloatArgType

__all__ = [
    "uniform_to_gaussian_quadprob",
    "sum_polynomials",
]


def uniform_to_gaussian_quadprob(
    quadprob: QuadratureProblem, mean: FloatArgType = 0.0, var: FloatArgType = 1.0
) -> QuadratureProblem:

    """Creates a new QuadratureProblem for integration against a Gaussian on R^d by
    using an existing QuadratureProblem whose integrand is suitable for integration
    against the Lebesgue measure on [0,1]^d.

    The multivariate Gaussian is of the form N(mean*(1,...,1),var^2 I_d).

    Using the change of variable formula, we have that

    .. math::  \int_{[0,1]^d} f(x) dx = \int_{\mathbb{R}^d} h(x) \phi(x) dx

    where :math:'h(x)=f(\Phi((x-mean)/var))', :math:'\phi(x)' is the Gaussian probability density function
    and :math:'\Phi(x)' an elementwise application of the Gaussian cummulative distribution function.

    Parameters
    ----------
        quadprob
            A QuadratureProblem instance which includes an integrand defined on [0,1]^d
        mean
            Mean of the Gaussian distribution.
        var
            Diagonal element for the covariance matrix of the Gaussian distribution.

    Returns
    -------
            A new Quadrature Problem instance with a transformed integrand taking inputs in :math:'\mathbb{R}^d'.

    Example
    -------
        uniform_to_gaussian_quadprob(genz_continuous(1))

    References
    ----------
        Si, S., Oates, C. J., Duncan, A. B., Carin, L. & Briol. F-X. (2021). Scalable control variates for Monte Carlo methods via stochastic optimization.
        Proceedings of the 14th Monte Carlo and Quasi-Monte Carlo Methods (MCQMC) conference 2020. arXiv:2006.07487.
    """

    dim = len(quadprob.lower_bd)

    # Check that the original quadrature problem was defined on [0,1]^d
    if np.any(quadprob.lower_bd != 0.0):
        raise ValueError(f"quadprob is not an integration problem over [0,1]^d")
    if np.any(quadprob.upper_bd != 1.0):
        raise ValueError(f"quadprob is not an integration problem over [0,1]^d")

    # Check that the original quadrature problem has a scalar valued solution
    if isinstance(quadprob.solution, float) is False:
        raise TypeError(f"The solution of quadprob is not a scalar.")

    # Construct transformation of the integrand
    def uniform_to_gaussian_integrand(
        func: Callable[[np.ndarray], np.ndarray],
        mean: FloatArgType = 0.0,
        var: FloatArgType = 1.0,
    ) -> Callable[[np.ndarray], np.ndarray]:

        # mean and var should be either one-dimensional, or an array of dimension d
        if isinstance(mean, float) is False:
            raise TypeError(f"The mean parameter should be a float.")

        if isinstance(var, float) is False or var <= 0.0:
            raise TypeError(f"The variance should be a positive float.")

        def newfunc(x):
            return func(norm.cdf((x - mean) / var))

        return newfunc

    return QuadratureProblem(
        integrand=uniform_to_gaussian_integrand(
            func=quadprob.integrand, mean=mean, var=var
        ),
        lower_bd=np.broadcast_to(-np.Inf, dim),
        upper_bd=np.broadcast_to(np.Inf, dim),
        output_dim=None,
        solution=quadprob.solution,
    )


def sum_polynomials(
    dim: int, a: np.ndarray = None, b: np.ndarray = None, var: FloatArgType = 1.0
) -> QuadratureProblem:
    """Quadrature problem with an integrand taking the form of a sum of polynomials over
    :math:'\mathbb{R}^d'.

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
