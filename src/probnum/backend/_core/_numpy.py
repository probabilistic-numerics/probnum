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
    sin,
    single,
    sqrt,
    sum,
    zeros,
    zeros_like,
)


def cast(a: np.ndarray, dtype=None, casting="unsafe", copy=None):
    return a.astype(dtype=dtype, casting=casting, copy=copy)


def is_floating(a: np.ndarray):
    return np.issubdtype(a.dtype, np.floating)


def to_numpy(a: np.ndarray) -> np.ndarray:
    return a


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f
