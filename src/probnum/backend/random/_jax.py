import functools
import secrets
from typing import Optional, Sequence, Tuple

import jax
from jax import numpy as jnp

from probnum.typing import DTypeArgType, FloatLike, ShapeLike


def seed(seed: Optional[int]) -> jnp.ndarray:
    if seed is None:
        seed = secrets.randbits(128)

    if not isinstance(seed, int):
        return seed

    return jax.random.PRNGKey(seed)


def split(seed: jnp.ndarray, num: int = 2) -> Sequence[jnp.ndarray]:
    return jax.random.split(key=seed, num=num)


def standard_normal(seed: jnp.ndarray, shape=(), dtype=jnp.double):
    return jax.random.normal(key=seed, shape=shape, dtype=dtype)


def gamma(
    seed: jnp.ndarray,
    shape_param: FloatLike,
    scale_param: FloatLike = 1.0,
    shape: ShapeLike = (),
    dtype: DTypeArgType = jnp.double,
):
    return (
        jax.random.gamma(key=seed, a=shape_param, shape=shape, dtype=dtype)
        * scale_param
    )


@functools.partial(jax.jit, static_argnames=("n", "shape", "dtype"))
def uniform_so_group(
    seed: jnp.ndarray,
    n: int,
    shape: ShapeLike = (),
    dtype: DTypeArgType = jnp.double,
) -> jnp.ndarray:
    if n == 1:
        return jnp.ones(shape + (1, 1), dtype=dtype)

    return _uniform_so_group_pushforward_fn(
        standard_normal(seed, shape=shape + (n - 1, n), dtype=dtype)
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

    row_norms_sq = jnp.sum(X ** 2, axis=1)

    X = X.at[jnp.diag_indices(n - 1)].set(jnp.sqrt(row_norms_sq) * D)
    X /= jnp.sqrt((row_norms_sq - X_diag ** 2 + jnp.diag(X) ** 2) / 2.0)[:, None]

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
