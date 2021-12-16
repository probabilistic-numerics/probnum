import numpy as np
from scipy import special

from probnum.typing import FloatArgType

__all__ = [
    "sum_polynomials",
]


def sum_polynomials(
    dim: int, a: np.ndarray = None, b: np.ndarray = None
) -> QuadratureProblem:
    """Integrand taking the form of a sum of polynomials over :math:'\mathbb{R}^d'.

    .. math::  f(x) = \sum_{j=0}^p \prod_{i=1}^dim a_{ji} x_i^{b_ji}

    Parameters
    ----------
        dim
            Dimension of the integration domain
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

    # Specify default values of parameters a and u
    if a is None:
        a = np.broadcast_to(1.0, dim)
    if b is None:
        b = np.broadcast_to(1, dim)

    # Check that the parameters have valid values and shape

    if a.shape != (dim,):
        raise ValueError(
            f"Invalid shape {a.shape} for parameter `a`. Expected {(dim,)}."
        )

    if b.shape != (dim,):
        raise ValueError(
            f"Invalid shape {b.shape} for parameter `b`. Expected {(dim,)}."
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
