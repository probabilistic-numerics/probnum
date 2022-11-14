"""JAX array manipulation functions."""
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp

from ..typing import ShapeType


def broadcast_arrays(*arrays: jnp.ndarray) -> List[jnp.ndarray]:
    return jnp.broadcast_arrays(*arrays)


def broadcast_to(x: jnp.ndarray, /, shape: ShapeType) -> jnp.ndarray:
    return jnp.broadcast_to(x, shape=shape)


def concat(
    arrays: Union[Tuple[jnp.ndarray, ...], List[jnp.ndarray]],
    /,
    *,
    axis: Optional[int] = 0,
) -> jnp.ndarray:
    return jnp.concatenate(arrays, axis=axis)


def expand_axes(x: jnp.ndarray, /, *, axis: int = 0) -> jnp.ndarray:
    return jnp.expand_dims(x, axis=axis)


def flip(
    x: jnp.ndarray, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> jnp.ndarray:
    return jnp.flip(x, axis=axis)


def permute_axes(x: jnp.ndarray, /, axes: Tuple[int, ...]) -> jnp.ndarray:
    return jnp.transpose(x, axes=axes)


def swap_axes(x: jnp.ndarray, /, axis1: int, axis2: int) -> jnp.ndarray:
    return jnp.swapaxes(x, axis1=axis1, axis2=axis2)


def reshape(
    x: jnp.ndarray, /, shape: ShapeType, *, copy: Optional[bool] = None
) -> jnp.ndarray:
    if copy is not None:
        if copy:
            out = jnp.copy(x)
    return jnp.reshape(out, newshape=shape)


def roll(
    x: jnp.ndarray,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> jnp.ndarray:
    return jnp.roll(x, shift=shift, axis=axis)


def squeeze(x: jnp.ndarray, /, axis: Union[int, Tuple[int, ...]]) -> jnp.ndarray:
    return jnp.squeeze(x, axis=axis)


def stack(
    arrays: Union[Tuple[jnp.ndarray, ...], List[jnp.ndarray]], /, *, axis: int = 0
) -> jnp.ndarray:
    return jnp.stack(arrays, axis=axis)


def hstack(arrays: Union[Tuple[jnp.ndarray, ...], List[jnp.ndarray]], /) -> jnp.ndarray:
    return jnp.hstack(arrays)


def vstack(arrays: Union[Tuple[jnp.ndarray, ...], List[jnp.ndarray]], /) -> jnp.ndarray:
    return jnp.vstack(arrays)
