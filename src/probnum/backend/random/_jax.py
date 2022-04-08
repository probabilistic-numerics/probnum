"""Functionality for random number generation implemented in the JAX backend."""
from __future__ import annotations

import functools
import secrets
from typing import Sequence

import jax
from jax import numpy as jnp

from probnum.backend.typing import DTypeLike, FloatLike, Seed, ShapeLike, ShapeType

RNGState = jax.random.PRNGKey


def rng_state(seed: Seed) -> RNGState:
    if seed is None:
        seed = secrets.randbits(128)

    if not isinstance(seed, int):
        return seed

    return jax.random.PRNGKey(seed)


def split(rng_state: RNGState, num: int = 2) -> Sequence[RNGState]:
    return jax.random.split(key=rng_state, num=num)


def uniform(
    rng_state: RNGState,
    shape: ShapeType = (),
    dtype: jnp.dtype = jnp.double,
    minval: jnp.ndarray = jnp.array(0.0),
    maxval: jnp.ndarray = jnp.array(1.0),
) -> jnp.ndarray:
    return jax.random.uniform(
        key=rng_state, shape=shape, dtype=dtype, minval=minval, maxval=maxval
    )


def standard_normal(
    rng_state: RNGState,
    shape: ShapeType = (),
    dtype: jnp.dtype = jnp.double,
) -> jnp.ndarray:
    return jax.random.normal(key=rng_state, shape=shape, dtype=dtype)


def gamma(
    rng_state: RNGState,
    shape_param: jnp.ndarray,
    scale_param: jnp.ndarray = jnp.array(1.0),
    shape: ShapeType = (),
    dtype: jnp.dtype = jnp.double,
) -> jnp.ndarray:
    return (
        jax.random.gamma(key=rng_state, a=shape_param, shape=shape, dtype=dtype)
        * scale_param
    )


@functools.partial(jax.jit, static_argnames=("n", "shape", "dtype"))
def uniform_so_group(
    rng_state: RNGState,
    n: int,
    shape: ShapeType = (),
    dtype: jnp.dtype = jnp.double,
) -> jnp.ndarray:
    if n == 1:
        return jnp.ones(shape + (1, 1), dtype=dtype)

    return _uniform_so_group_pushforward_fn(
        standard_normal(rng_state, shape=shape + (n - 1, n), dtype=dtype)
    )


@functools.partial(jnp.vectorize, signature="(M,N)->(N,N)")
def _uniform_so_group_pushforward_fn(omega: jnp.ndarray) -> jnp.ndarray:
    n = omega.shape[1]

    assert omega.shape == (n - 1, n)

    X = jnp.triu(omega)

    X_diag = jnp.diag(X)
    D = jnp.vectorize(
        lambda x: jax.lax.cond(
            x != 0,
            lambda x: jnp.sign(x),
            lambda _: jnp.ones((), dtype=omega.dtype),
            x,
        ),
    )(X_diag)

    row_norms_sq = jnp.sum(X**2, axis=1)

    X = X.at[jnp.diag_indices(n - 1)].set(jnp.sqrt(row_norms_sq) * D)
    X /= jnp.sqrt((row_norms_sq - X_diag**2 + jnp.diag(X) ** 2) / 2.0)[:, None]

    H = jax.lax.fori_loop(
        lower=0,
        upper=n - 1,
        body_fun=lambda idx, H: H - jnp.outer(H @ X[idx, :], X[idx, :]),
        init_val=jnp.eye(n, dtype=omega.dtype),
    )

    D = jnp.append(
        D,
        (-1.0 if n % 2 == 0 else 1.0) * jnp.prod(D),
    )

    return D[:, None] * H
