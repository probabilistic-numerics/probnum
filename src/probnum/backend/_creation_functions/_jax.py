"""JAX array creation functions."""
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax.numpy import tril, triu  # pylint: disable=redefined-builtin, unused-import


def asarray(
    obj: Union[
        jnp.ndarray, bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype: Optional["probnum.backend.Dtype"] = None,
    device: Optional["probnum.backend.Device"] = None,
    copy: Optional[bool] = None,
) -> jnp.ndarray:
    if copy is None:
        copy = True
    x = jnp.array(obj, dtype=dtype, copy=copy)
    if device is not None:
        return jax.device_put(x, device=device)
    return x
