"""JAX array manipulation functions."""
from typing import Optional

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy import (  # pylint: disable=unused-import
        atleast_1d,
        atleast_2d,
        broadcast_arrays,
        broadcast_shapes,
        broadcast_to,
        concatenate as concat,
        expand_dims as expand_axes,
        flip,
        hstack,
        moveaxis as move_axes,
        roll,
        squeeze,
        stack,
        swapaxes as swap_axes,
        tile,
        transpose as permute_axes,
        vstack,
    )
except ModuleNotFoundError:
    pass

from ..typing import ShapeType


def reshape(
    x: "jax.Array", /, shape: ShapeType, *, copy: Optional[bool] = None
) -> "jax.Array":
    if copy is not None:
        if copy:
            out = jnp.copy(x)
    return jnp.reshape(out, newshape=shape)
