"""Implementation of linear algebra functionality in JAX."""

import functools
from typing import Literal, Optional, Tuple, Union

import jax
from jax import numpy as jnp
from jax.numpy import diagonal  # pylint: disable=unused-import
from jax.numpy.linalg import eigh, eigvalsh, solve, svd  # pylint: disable=unused-import


def vector_norm(
    x: jnp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal["inf", "-inf"]] = 2,
) -> jnp.ndarray:
    return jnp.linalg.norm(x=x, ord=ord, keepdims=keepdims, axis=axis)


def matrix_norm(x: jnp.ndarray, /, *, keepdims: bool = False, ord="fro") -> jnp.ndarray:
    return jnp.linalg.norm(x=x, ord=ord, keepdims=keepdims, axis=(-2, -1))


def cholesky(x: jnp.ndarray, /, *, upper: bool = False) -> jnp.ndarray:
    L = jax.numpy.linalg.cholesky(x)

    return jnp.conj(L.swapaxes(-2, -1)) if upper else L


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


def qr(
    x: jnp.ndarray, /, *, mode: Literal["reduced", "complete", "r"] = "reduced"
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if mode == "r":
        r = jnp.linalg.qr(x, mode=mode)
        q = None
    else:
        q, r = jnp.linalg.qr(x, mode=mode)

    return q, r
