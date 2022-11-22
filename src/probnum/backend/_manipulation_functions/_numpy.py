"""NumPy array manipulation functions."""

from typing import Optional

import numpy as np
from numpy import (  # pylint: disable=unused-import
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

from ..typing import ShapeType


def reshape(
    x: np.ndarray, /, shape: ShapeType, *, copy: Optional[bool] = None
) -> np.ndarray:
    if copy is not None:
        if copy:
            out = np.copy(x)
    return np.reshape(out, newshape=shape)
