"""Utility functions for square-root Gaussian filtering and smoothing.

They provide the backend functionality for the methods in SquareRootKalman.


See
    https://arxiv.org/pdf/1610.04397.pdf
and
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf
for information.
"""

import numpy as np
import scipy.linalg


# used for predict() and measure(), but more general than that,
# so it has a more general name than the functions below.
def cholesky_update(S1, S2=None):
    """Compute Cholesky update/factorization :math:`C C^\top = S_1 S_1^\top + S_2 S_2^\top`.

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


def sqrt_kalman_update(measmat, meascov_cholesky, predcov_cholesky):
    """Computes the Kalman update in square-root form.

    See Eq. 48 in
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf.

    Returns
    -------
    measured_cholesky
        Cholesky factor of the covariance of "the measured random variable".
    kalman_gain
        Kalman gain.
    postcov_cholesky
        Cholesky factor of the posterior covariance (i.e. after the update).
    """
    zeros = np.zeros(measmat.T.shape)
    blockmat = np.block(
        [[meascov_cholesky, measmat @ predcov_cholesky], [zeros, predcov_cholesky]]
    ).T
    big_triu = np.linalg.qr(blockmat, mode="r")

    meas_dim = len(measmat)
    measured_triu = big_triu[:meas_dim, :meas_dim]
    measured_cholesky = triu_to_positive_tril(measured_triu)
    postcov_triu = big_triu[meas_dim:, meas_dim:].T
    postcov_cholesky = triu_to_positive_tril(postcov_triu)
    kalman_gain = scipy.linalg.cho_solve(
        (measured_cholesky, True), big_triu[:meas_dim, meas_dim:]
    ).T
    return measured_cholesky, kalman_gain, postcov_cholesky


def sqrt_smoothing_step(
    sqrtm_unsmoothed_cov_past,
    dynamicsmat,
    diffmat_cholesky,
    sqrtm_smoothed_cov_future,
    smoothing_gain,
):
    """Smoothing step in square-root form.

    See Eq. 45 in
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf.
    """
    dim = len(dynamicsmat)
    zeros = np.zeros((dim, dim))
    blockmat = np.block(
        [
            [sqrtm_unsmoothed_cov_past.T @ dynamicsmat.T, sqrtm_unsmoothed_cov_past.T],
            [diffmat_cholesky.T, zeros],
            [zeros, sqrtm_smoothed_cov_future.T @ smoothing_gain.T],
        ]
    )
    big_triu = np.linalg.qr(blockmat, mode="r")
    chol_unsmoothed_cov_past = big_triu[dim : 2 * dim, dim:]
    return triu_to_positive_tril(chol_unsmoothed_cov_past)


def triu_to_positive_tril(triu_mat):
    """Change an upper triangular matrix into a valid lower Cholesky factor.

    Transpose and change the sign of the diagonals to '+' if necessary.
    """
    tril_mat = triu_mat.T
    with_pos_diag = tril_mat @ np.diag(np.sign(np.diag(tril_mat)))
    return with_pos_diag
