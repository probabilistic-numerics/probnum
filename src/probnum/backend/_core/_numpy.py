from typing import Tuple, Union

import numpy as np
from numpy import (  # pylint: disable=redefined-builtin, unused-import
    abs,
    all,
    any,
    arange,
    atleast_1d,
    atleast_2d,
    broadcast_arrays,
    broadcast_shapes,
    broadcast_to,
    concatenate,
    diag,
    diagonal,
    dtype as asdtype,
    einsum,
    exp,
    eye,
    finfo,
    flip,
    full,
    full_like,
    hstack,
    isfinite,
    isnan,
    linspace,
    log,
    max,
    maximum,
    meshgrid,
    minimum,
    moveaxis,
    ndim,
    ones,
    ones_like,
    sign,
    sin,
    sqrt,
    squeeze,
    stack,
    sum,
    swapaxes,
    tile,
    vectorize,
    vstack,
    zeros,
    zeros_like,
)


def to_numpy(*arrays: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if len(arrays) == 1:
        return arrays[0]

    return tuple(arrays)


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f
