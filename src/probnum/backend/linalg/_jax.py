import functools

import jax
from jax.numpy.linalg import norm, qr, svd
from jax.scipy.linalg import cholesky


@functools.partial(jax.jit, static_argnames=("transpose", "lower", "unit_diagonal"))
def solve_triangular(
    A: jax.numpy.ndarray,
    b: jax.numpy.ndarray,
    *,
    transpose: bool = False,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> jax.numpy.ndarray:
    if b.ndim in (1, 2):
        return jax.scipy.linalg.solve_triangular(
            A,
            b,
            trans=1 if transpose else 0,
            lower=lower,
            unit_diagonal=unit_diagonal,
        )

    @functools.partial(jax.numpy.vectorize, signature="(n,n),(n,k)->(n,k)")
    def _solve_triangular_vectorized(
        A: jax.numpy.ndarray,
        b: jax.numpy.ndarray,
    ) -> jax.numpy.ndarray:
        return jax.scipy.linalg.solve_triangular(
            A,
            b,
            trans=1 if transpose else 0,
            lower=lower,
            unit_diagonal=unit_diagonal,
        )

    return _solve_triangular_vectorized(A, b)


@functools.partial(jax.jit, static_argnames=("lower", "overwrite_b", "check_finite"))
def solve_cholesky(
    cholesky: jax.numpy.ndarray,
    b: jax.numpy.ndarray,
    *,
    lower: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
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
