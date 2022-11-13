"""Searching functions on JAX arrays."""
from typing import Optional

import jax.numpy as jnp
from jax.numpy import nonzero, where  # pylint: disable=redefined-builtin, unused-import


def argmax(
    x: jnp.ndarray, /, *, axis: Optional[int] = None, keepdims: bool = False
) -> jnp.ndarray:
    return jnp.argmax(a=x, axis=axis, keepdims=keepdims)


def argmin(
    x: jnp.ndarray, /, *, axis: Optional[int] = None, keepdims: bool = False
) -> jnp.ndarray:
    return jnp.argmin(a=x, axis=axis, keepdims=keepdims)
