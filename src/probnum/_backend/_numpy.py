from numpy import (
    array,
    asarray,
    atleast_1d,
    atleast_2d,
    broadcast_arrays,
    broadcast_shapes,
    exp,
    ndim,
    ones_like,
    sqrt,
    sum,
    zeros,
    zeros_like,
)


def jit(f):
    return f


def grad(*args, **kwargs):
    raise NotImplementedError()
