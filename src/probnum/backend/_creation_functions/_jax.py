"""JAX array creation functions."""
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax.numpy import (  # pylint: disable=redefined-builtin, unused-import
    arange,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)


def asarray(
    obj: Union[
        jnp.ndarray, bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype=None,
    device=None,
    copy: Optional[bool] = None,
) -> jnp.ndarray:
    if copy is None:
        copy = True
    x = jnp.array(obj, dtype=dtype, copy=copy)
    if device is not None:
        return jax.device_put(x, device=device)
    return x
