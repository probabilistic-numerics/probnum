"""Utility functions for square-root Gaussian filtering and smoothing.

See
    https://arxiv.org/pdf/1610.04397.pdf
and
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.1059&rep=rep1&type=pdf
for information.
"""

import numpy as np


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


def sqrt_kalman_update(L_R, H, L_C):

    zeros = np.zeros(H.T.shape)
    blockmat = np.block(((L_R, H @ L_C), (zeros, L_C))).T
    big_R = np.linalg.qr(blockmat, mode="r")

    size_small = len(H)
    R_S = big_R[:size_small, :size_small].T
    L_S = triu_to_positive_tril(R_S)
    R_P = big_R[size_small:, size_small:].T
    L_P = triu_to_positive_tril(R_P)
    K = np.linalg.solve(L_S, big_R[:size_small, size_small:]).T
    return L_S, K, L_P


def sqrt_smoothing_step(L_P_unsmoothed_past, A, L_Q, L_P_smoothed_fut, G):

    zeros = np.zeros(A.shape)
    blockmat = np.block(
        (
            (L_P_unsmoothed_past.T @ A.T, L_P_unsmoothed_past.T),
            (L_Q.T, zeros),
            (zeros, L_P_smoothed_fut.T @ G.T),
        )
    )
    big_R = np.linalg.qr(blockmat, mode="r")

    size_small = len(A)

    L_P_smoothed = big_R[size_small : 2 * size_small, size_small:]

    return L_P_smoothed


def triu_to_positive_tril(triu_mat):
    """Change an upper triangular matrix into a valid lower Cholesky factor.

    Transpose and change the sign of the diagonals to '+' if necessary.
    """
    tril_mat = triu_mat.T
    with_pos_diag = tril_mat @ np.diag(np.sign(np.diag(tril_mat)))
    return with_pos_diag
