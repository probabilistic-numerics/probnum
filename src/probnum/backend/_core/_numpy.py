from typing import Tuple, Union

import numpy as np
from numpy import (  # pylint: disable=redefined-builtin, unused-import
    abs,
    all,
    any,
    arange,
    array,
    atleast_1d,
    atleast_2d,
    bool_ as bool,
    broadcast_arrays,
    broadcast_shapes,
    broadcast_to,
    cdouble,
    concatenate,
    csingle,
    diag,
    diagonal,
    double,
    dtype as asdtype,
    einsum,
    exp,
    expand_dims,
    eye,
    finfo,
    flip,
    full,
    full_like,
    hstack,
    inf,
    int32,
    int64,
    isfinite,
    isnan,
    kron,
    linspace,
    log,
    max,
    maximum,
    meshgrid,
    moveaxis,
    ndim,
    ones,
    ones_like,
    pi,
    promote_types,
    reshape,
    result_type,
    sign,
    sin,
    single,
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


def cast(a: np.ndarray, dtype=None, casting="unsafe", copy=None):
    return a.astype(dtype=dtype, casting=casting, copy=copy)


def is_floating(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.floating)


def is_floating_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.floating)


def to_numpy(*arrays: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if len(arrays) == 1:
        return arrays[0]

    return tuple(arrays)


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f