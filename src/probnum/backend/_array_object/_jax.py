"""Array object in JAX."""
from typing import Tuple, Union

import jax.numpy as jnp
from jax.numpy import (  # pylint: disable=redefined-builtin, unused-import
    ndarray as Array,
    ndarray as Scalar,
    ndim,
)
from jaxlib.xla_extension import Device


def to_numpy(*arrays: jnp.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if len(arrays) == 1:
        return np.array(arrays[0])

    return tuple(np.array(arr) for arr in arrays)
