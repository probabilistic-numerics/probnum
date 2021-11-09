import numpy as np
import scipy.linalg
from scipy.linalg import cholesky


def cholesky_solve(
    cholesky: np.ndarray,
    b: np.ndarray,
    *,
    lower: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
):
    if b.ndim == 1:
        return scipy.linalg.cho_solve(
            (cholesky, lower),
            b,
            overwrite_b=overwrite_b,
            check_finite=check_finite,
        )

    b = b.transpose((-2,) + tuple(range(b.ndim - 2)) + (-1,))

    x = scipy.linalg.cho_solve(
        (cholesky, lower),
        b,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )

    return x.transpose(tuple(range(1, b.ndim - 1)) + (0, -1))
