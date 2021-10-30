import numpy as np
from numpy import (
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
    int32,
    int64,
    linspace,
    log,
    ndarray,
    ndim,
    ones,
    ones_like,
    pi,
    promote_types,
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


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f


def grad(*args, **kwargs):
    raise NotImplementedError()
