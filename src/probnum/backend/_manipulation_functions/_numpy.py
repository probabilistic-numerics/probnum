"""NumPy array manipulation functions."""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import atleast_1d, atleast_2d  # pylint: disable=unused-import

from ..typing import ShapeType


def broadcast_arrays(*arrays: np.ndarray) -> List[np.ndarray]:
    return np.broadcast_arrays(*arrays)


def broadcast_to(x: np.ndarray, /, shape: ShapeType) -> np.ndarray:
    return np.broadcast_to(x, shape=shape)


def concat(
    arrays: Union[Tuple[np.ndarray, ...], List[np.ndarray]],
    /,
    *,
    axis: Optional[int] = 0,
) -> np.ndarray:
    return np.concatenate(arrays, axis=axis)


def expand_axes(x: np.ndarray, /, *, axis: int = 0) -> np.ndarray:
    return np.expand_dims(x, axis=axis)


def flip(
    x: np.ndarray, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> np.ndarray:
    return np.flip(x, axis=axis)


def move_axes(
    x: np.ndarray,
    /,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> np.ndarray:
    return np.moveaxis(x, source, destination)


def permute_axes(x: np.ndarray, /, axes: Tuple[int, ...]) -> np.ndarray:
    return np.transpose(x, axes=axes)


def swap_axes(x: np.ndarray, /, axis1: int, axis2: int) -> np.ndarray:
    return np.swapaxes(x, axis1=axis1, axis2=axis2)


def reshape(
    x: np.ndarray, /, shape: ShapeType, *, copy: Optional[bool] = None
) -> np.ndarray:
    if copy is not None:
        if copy:
            out = np.copy(x)
    return np.reshape(out, newshape=shape)


def roll(
    x: np.ndarray,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> np.ndarray:
    return np.roll(x, shift=shift, axis=axis)


def squeeze(x: np.ndarray, /, axis: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return np.squeeze(x, axis=axis)


def stack(
    arrays: Union[Tuple[np.ndarray, ...], List[np.ndarray]], /, *, axis: int = 0
) -> np.ndarray:
    return np.stack(arrays, axis=axis)


def hstack(arrays: Union[Tuple[np.ndarray, ...], List[np.ndarray]], /) -> np.ndarray:
    return np.hstack(arrays)


def vstack(arrays: Union[Tuple[np.ndarray, ...], List[np.ndarray]], /) -> np.ndarray:
    return np.vstack(arrays)


def tile(A: np.ndarray, /, reps: ShapeType) -> np.ndarray:
    return np.tile(A, reps)
