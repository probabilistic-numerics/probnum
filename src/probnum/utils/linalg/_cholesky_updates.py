"""Cholesky updates."""


import typing

import numpy as np

__all__ = ["cholesky_update", "tril_to_positive_tril"]


def cholesky_update(
    S1: np.ndarray, S2: typing.Optional[np.ndarray] = None
) -> np.ndarray:
    r"""Compute Cholesky update/factorization :math:`L` such that :math:`L L^\top = S_1 S_1^\top + S_2 S_2^\top` holds.

    This can be used in various ways.
    For example, :math:`S_1` and :math:`S_2` do not need to be Cholesky factors; any matrix square-root is sufficient.
    As long as :math:`L L^\top = S_1 S_1^\top + S_2 S_2^\top` is well-defined (and admits a Cholesky-decomposition),
    :math:`S_1` and :math:`S_2` do not even have to be square.


    Parameters
    ----------
    S1 :
        First matrix square-root. Not necessarily a Cholesky factor, any (possibly even non-square) matrix :math:`S` such that :math:`C = S S^\top` holds, is sufficient.
    S2 :
        Second matrix square-root. Not necessarily a Cholesky factor, any (possibly even non-square) matrix :math:`S` such that :math:`C = S S^\top` holds, is sufficient.
        Optional. Default is None.

    Returns
    -------
    Lower Cholesky factor :math:`L` of :math:`L L^\top =S1 S1^\top + S2 S2^\top`, if ``S2`` was not None. Otherwise, lower Cholesky factor of :math:`L L^\top =S1 S1^\top`.


    Examples
    --------

    >>> from probnum.utils.linalg import cholesky_update
    >>> from probnum.problems.zoo.linalg import random_spd_matrix
    >>> import numpy as np

    Compute the Cholesky-factor of a sum of SPD matrices.

    >>> rng = np.random.default_rng(seed=3)
    >>> C1 = random_spd_matrix(rng, dim=5)
    >>> S1 = np.linalg.cholesky(C1)
    >>> C2 = random_spd_matrix(rng, dim=5)
    >>> S2 = np.linalg.cholesky(C2)
    >>> C = C1 + C2
    >>> S = cholesky_update(S1, S2)
    >>> np.allclose(np.linalg.cholesky(C), S)
    True

    Turn a (potentially non-square) matrix square-root into a Cholesky factor

    >>> A = np.random.rand(3, 5)
    >>> S = cholesky_update(A @ S1)
    >>> np.allclose(np.linalg.cholesky(A @ C1 @ A.T), S)
    True
    """
    if S2 is not None:
        stacked_up = np.vstack((S1.T, S2.T))
    else:
        stacked_up = np.vstack(S1.T)
    upper_sqrtm = np.linalg.qr(stacked_up, mode="r")
    lower_sqrtm = upper_sqrtm.T
    return tril_to_positive_tril(lower_sqrtm)


def tril_to_positive_tril(tril_mat: np.ndarray) -> np.ndarray:
    r"""Orthogonally transform a lower-triangular matrix into a lower-triangular matrix with positive diagonal.

    In other words, make it a valid lower Cholesky factor.

    The name of the function is based on `np.tril`.
    """
    d = np.sign(np.diag(tril_mat))

    # Numpy assigns sign 0 to 0.0, which eliminate entire rows in the operation below.
    d[d == 0] = 1.0

    # Fast(er) multiplication with a diagonal matrix from the right via broadcasting.
    with_pos_diag = tril_mat * d[None, :]
    return with_pos_diag
