"""Statistical functions implemented in JAX."""

from typing import Optional, Tuple, Union

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy import mean, prod, sum  # pylint: disable=unused-import
except ModuleNotFoundError:
    pass


def max(
    x: "jax.Array",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "jax.Array":
    return jnp.amax(x, axis=axis, keepdims=keepdims)


def min(
    x: "jax.Array",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> "jax.Array":
    return jnp.amin(x, axis=axis, keepdims=keepdims)


def std(
    x: "jax.Array",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> "jax.Array":
    return jnp.std(x, axis=axis, ddof=correction, keepdims=keepdims)


def var(
    x: "jax.Array",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> "jax.Array":
    return jnp.var(x, axis=axis, ddof=correction, keepdims=keepdims)
