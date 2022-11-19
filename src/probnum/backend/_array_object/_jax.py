"""Array object in JAX."""
from typing import Tuple, Union

try:
    # pylint: disable=redefined-builtin, unused-import
    import jax
    from jax import Array, Array as Scalar
    import jax.numpy as jnp
    from jax.numpy import ndim
    from jaxlib.xla_extension import Device
except ModuleNotFoundError:
    pass


def to_numpy(*arrays: "jax.Array") -> Union["jax.Array", Tuple["jax.Array", ...]]:
    if len(arrays) == 1:
        return jnp.array(arrays[0])

    return tuple(jnp.array(arr) for arr in arrays)
