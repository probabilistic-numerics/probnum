import typing

import numpy as np

import probnum.random_variables as pnrv


def condition_state_on_measurement(attained_rv, forwarded_rv, rv, gain):
    new_mean = rv.mean + gain @ (attained_rv.mean - forwarded_rv.mean)
    new_cov = rv.cov + gain @ (attained_rv.cov - forwarded_rv.cov) @ gain.T

    return pnrv.Normal(new_mean, new_cov)


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
        stacked_up = np.vstack(S1.T)
    upper_sqrtm = np.linalg.qr(stacked_up, mode="r")
    return triu_to_positive_tril(upper_sqrtm)


def triu_to_positive_tril(triu_mat: np.ndarray) -> np.ndarray:
    r"""Change an upper triangular matrix into a valid lower Cholesky factor.

    Transpose, and change the sign of the diagonals to '+' if necessary.
    The name of the function is leaned on `np.triu` and `np.tril`.
    """
    tril_mat = triu_mat.T
    with_pos_diag = tril_mat @ np.diag(np.sign(np.diag(tril_mat)))
    return with_pos_diag
