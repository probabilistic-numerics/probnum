"""Array object in JAX."""
from typing import Tuple, Union

import jax.numpy as jnp
from jax.numpy import (  # pylint: disable=redefined-builtin, unused-import
    ndarray as Array,
    ndarray as Scalar,
    ndim,
)
from jaxlib.xla_extension import Device


def to_numpy(*arrays: jnp.ndarray) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    if len(arrays) == 1:
        return jnp.array(arrays[0])

    return tuple(jnp.array(arr) for arr in arrays)
