"""Cholesky updates."""
import typing
import numpy as np

__all__ = ["cholesky_update", "tril_to_positive_tril"]


def cholesky_update(
    S1: np.ndarray, S2: typing.Optional[np.ndarray] = None
) -> np.ndarray:
    r"""Compute Cholesky factorization from two matrices S1 and S2.

    Parameters
    ----------
    S1 :
       First matrix, a Hermitian matrix with positive eigenvalues.
    S2 :
        Second matrix, a Hermitian matrix with positive eigenvalues.

    Returns
    -------
    Lower Cholesky factor
    """
    if S2 is not None:
        stacked_up = np.vstack((S1.T, S2.T))
    else:
        stacked_up = np.vstack(S1.T)
    upper_sqrtm = np.linalg.qr(stacked_up, mode="r")
    if S1.ndim == 1:
        lower_sqrtm = upper_sqrtm.T
    elif S1.shape[0] <= S1.shape[1]:
        lower_sqrtm = upper_sqrtm.T
    else:
        lower_sqrtm = np.zeros((S1.shape[0], S1.shape[0]))
        lower_sqrtm[:, : -(S1.shape[0] - S1.shape[1])] = upper_sqrtm.T

    return tril_to_positive_tril(lower_sqrtm)


def tril_to_positive_tril(tril_mat: np.ndarray) -> np.ndarray:
    r"""Orthogonally transform a lower-triangular matrix into a lower-triangular matrix with positive diagonal.

    Parameters:
    --------

    tril_mat:
    A lower triangular matrix

    returns:
    --------
    A matrix
    """
    d = np.sign(np.diag(tril_mat))

    # Numpy assigns sign 0 to 0.0, which eliminate entire rows in the operation below.
    d[d == 0] = 1.0

    # Fast(er) multiplication with a diagonal matrix from the right via broadcasting.
    with_pos_diag = tril_mat * d[None, :]
    return with_pos_diag
