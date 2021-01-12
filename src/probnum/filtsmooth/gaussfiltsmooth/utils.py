"""Utility functions for Gaussian filtering and smoothing."""

import numpy as np


def cholesky_update(S1, S2=None):
    """Compute Cholesky factorization :math:`C C^\top = S_1 S_1^\top + S_2 S_2^\top`.

    This can be used in various ways.
    For one, :math:`S_1` and :math:`S_2` do not need to be Cholesky factors; any matrix square-root is sufficient.
    As long as :math:`C C^\top = S_1 S_1^\top + S_2 S_2^\top` is well-defined (and admits a Cholesky-decomposition),
    :math:`S_1` and :math:`S_2` do not even have to be square.
    """
    # doc might need a bit more explanation in the future
    if S2 is not None:
        stacked_up = np.vstack((S1.T, S2.T))
    else:
        stacked_up = np.vstack((S1.T))
    lower_sqrtm = np.linalg.qr(stacked_up, mode="r").T
    with_pos_diag = lower_sqrtm @ np.diag(np.sign(np.diag(lower_sqrtm)))
    return with_pos_diag
