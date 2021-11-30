import functools

import jax
from jax.scipy.linalg import cholesky


@functools.partial(jax.jit, static_argnames=("lower", "overwrite_b", "check_finite"))
def cholesky_solve(
    cholesky: jax.numpy.ndarray,
    b: jax.numpy.ndarray,
    *,
    lower: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True
):
    @functools.partial(jax.numpy.vectorize, signature="(n,n),(n,k)->(n,k)")
    def _cho_solve_vectorized(
        cholesky: jax.numpy.ndarray,
        b: jax.numpy.ndarray,
    ):
        return jax.scipy.linalg.cho_solve(
            (cholesky, lower),
            b,
            overwrite_b=overwrite_b,
            check_finite=check_finite,
        )

    if b.ndim == 1:
        return _cho_solve_vectorized(
            cholesky,
            b[:, None],
        )[:, 0]

    return _cho_solve_vectorized(cholesky, b)
