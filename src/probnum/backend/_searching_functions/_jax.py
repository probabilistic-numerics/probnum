"""Searching functions on JAX arrays."""
from typing import Optional

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy import (  # pylint: disable=redefined-builtin, unused-import
        nonzero,
        where,
    )
except ModuleNotFoundError:
    pass


def argmax(
    x: "jax.Array", /, *, axis: Optional[int] = None, keepdims: bool = False
) -> "jax.Array":
    return jnp.argmax(a=x, axis=axis, keepdims=keepdims)


def argmin(
    x: "jax.Array", /, *, axis: Optional[int] = None, keepdims: bool = False
) -> "jax.Array":
    return jnp.argmin(a=x, axis=axis, keepdims=keepdims)
