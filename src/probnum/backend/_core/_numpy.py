from typing import Tuple

import numpy as np
from numpy import (  # pylint: disable=redefined-builtin, unused-import
    all,
    array,
    asarray,
    atleast_1d,
    atleast_2d,
    bool_ as bool,
    broadcast_arrays,
    broadcast_shapes,
    cdouble,
    csingle,
    diag,
    double,
    dtype,
    dtype as asdtype,
    exp,
    eye,
    finfo,
    full,
    full_like,
    inf,
    int32,
    int64,
    isfinite,
    linspace,
    log,
    maximum,
    ndarray,
    ndim,
    ones,
    ones_like,
    pi,
    promote_types,
    reshape,
    sin,
    single,
    sqrt,
    sum,
    swapaxes,
    zeros,
    zeros_like,
)


def cast(a: np.ndarray, dtype=None, casting="unsafe", copy=None):
    return a.astype(dtype=dtype, casting=casting, copy=copy)


def is_floating(a: np.ndarray) -> bool:
    return np.issubdtype(a.dtype, np.floating)


def is_floating_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.floating)


def to_numpy(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(arrays)


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f
