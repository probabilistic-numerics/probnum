"""Statistical functions implemented in JAX."""

from typing import Optional, Tuple, Union

import jax.numpy as jnp


def max(
    x: jnp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    return jnp.amax(x, axis=axis, keepdims=keepdims)


def min(
    x: jnp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    return jnp.amin(x, axis=axis, keepdims=keepdims)


def mean(
    x: jnp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    return jnp.mean(x, axis=axis, keepdims=keepdims)


def prod(
    x: jnp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[jnp.dtype] = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    return jnp.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)


def sum(
    x: jnp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[jnp.dtype] = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    return jnp.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)


def std(
    x: jnp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> jnp.ndarray:
    return jnp.std(x, axis=axis, ddof=correction, keepdims=keepdims)


def var(
    x: jnp.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> jnp.ndarray:
    return jnp.var(x, axis=axis, ddof=correction, keepdims=keepdims)
