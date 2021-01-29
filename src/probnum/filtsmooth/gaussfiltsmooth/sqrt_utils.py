"""Utility functions for square-root Gaussian filtering and smoothing.

They provide the backend functionality for the methods in SquareRootKalman.


See
    https://arxiv.org/pdf/1610.04397.pdf
and
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf
for information.


The functions in here are intendend to be rather low-level, which is why they
take means and covariances as arguments, not random variables, etc..

Matrices whose name starts with a capital `S`, i.e. `SQ`, or `SQ` are square-roots.
The short names are chosen, because the matrix-formula heavy functions seem more readable with super short names.
"""


import typing

import numpy as np


# used for predict() and measure(), but more general than that,
# so it has a more general name than the functions below.
def cholesky_update(
    S1: np.ndarray, S2: typing.Optional[np.ndarray] = None
) -> np.ndarray:
    r"""Compute Cholesky update/factorization :math:`C C^\top = S_1 S_1^\top + S_2 S_2^\top`.

    This can be used in various ways.
    For example, :math:`S_1` and :math:`S_2` do not need to be Cholesky factors; any matrix square-root is sufficient.
    As long as :math:`C C^\top = S_1 S_1^\top + S_2 S_2^\top` is well-defined (and admits a Cholesky-decomposition),
    :math:`S_1` and :math:`S_2` do not even have to be square.
    """
    # doc might need a bit more explanation in the future
    # perhaps some doctest or so?
    if S2 is not None:
        stacked_up = np.vstack((S1.T, S2.T))
    else:
        stacked_up = np.vstack((S1.T))
    upper_sqrtm = np.linalg.qr(stacked_up, mode="r")
    return triu_to_positive_tril(upper_sqrtm)


# Perhaps `SR` can be made optional, too...
def sqrt_kalman_update(
    H: np.ndarray, SR: np.ndarray, SC: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    r"""Compute the Kalman update in square-root form.

    Assumes a measurement model of the form

        .. math::  x \mapsto N(H x, R)

    and acts only on the square-root of the predicted covariance.

    See Eq. 48 in
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf.

    Parameters
    ----------
    H
        Linear(ised) observation matrix.
    SR
        Matrix square-root of the measurement diffusion matrix math:`R`, :math:`R=\sqrt{R} \sqrt{R}^\top`.
        Can, but does not have to be a Cholesky factor.
    SC
        Matrix square-root of the predicted covariance math:`C`, :math:`C=\sqrt{C} \sqrt{C}^\top`.
        Can, but does not have to be a Cholesky factor.

    Returns
    -------
    measured_cholesky
        Cholesky factor of the covariance of "the measured random variable".
    kalman_gain
        Kalman gain.
    postcov_cholesky
        Cholesky factor of the posterior covariance (i.e. after the update).
    """
    zeros = np.zeros(H.T.shape)
    blockmat = np.block([[SR, H @ SC], [zeros, SC]]).T
    big_triu = np.linalg.qr(blockmat, mode="r")
    ndim_measurements = len(H)

    measured_triu = big_triu[:ndim_measurements, :ndim_measurements]
    measured_cholesky = triu_to_positive_tril(measured_triu)

    postcov_triu = big_triu[ndim_measurements:, ndim_measurements:]
    postcov_cholesky = triu_to_positive_tril(postcov_triu)
    kalman_gain = big_triu[:ndim_measurements, ndim_measurements:].T @ np.linalg.inv(
        measured_triu.T
    )

    return measured_cholesky, kalman_gain, postcov_cholesky


def sqrt_smoothing_step(
    SC_past: np.ndarray,
    A: np.ndarray,
    SQ: np.ndarray,
    SC_futu: np.ndarray,
    G: np.ndarray,
) -> np.ndarray:
    r"""Smoothing step in square-root form.

    Assumes a prior dynamic model of the form

        .. math:: x \\mapsto N(A x, Q).

    For the mathematical justification of this step, see Eq. 45 in
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf.

    Parameters
    ----------
    SC_past
        Square root of the filtered (not yet smoothed) covariance at time :math:`t_n`.
    A
        Dynamics matrix :math:`A`.
    SQ
        Square root of the diffusion matrix :math:`Q`, `Q=SQ SQ.T`.
    SC_futu
        Square root of the smoothed covariance at time :math:`t_{n+1}`.
    G
        Smoothing gain.
    """
    dim = len(A)
    zeros = np.zeros((dim, dim))
    blockmat = np.block(
        [
            [SC_past.T @ A.T, SC_past.T],
            [SQ.T, zeros],
            [zeros, SC_futu.T @ G.T],
        ]
    )
    big_triu = np.linalg.qr(blockmat, mode="r")
    SC = big_triu[dim : 2 * dim, dim:]
    return triu_to_positive_tril(SC)


def triu_to_positive_tril(triu_mat: np.ndarray) -> np.ndarray:
    r"""Change an upper triangular matrix into a valid lower Cholesky factor.

    Transpose, and change the sign of the diagonals to '+' if necessary.
    """
    tril_mat = triu_mat.T
    with_pos_diag = tril_mat @ np.diag(np.sign(np.diag(tril_mat)))
    return with_pos_diag
