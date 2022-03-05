from typing import Tuple, Union

import jax
from jax.numpy import (  # pylint: disable=redefined-builtin, unused-import
    abs,
    all,
    arange,
    array,
    asarray,
    atleast_1d,
    atleast_2d,
    bool_ as bool,
    broadcast_arrays,
    broadcast_shapes,
    cdouble,
    complex64 as csingle,
    concatenate,
    diag,
    diagonal,
    double,
    dtype,
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
    linspace,
    log,
    maximum,
    meshgrid,
    moveaxis,
    ndarray,
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
    vstack,
    zeros,
    zeros_like,
)
import numpy as np

jax.config.update("jax_enable_x64", True)


def broadcast_to(
    array: jax.numpy.ndarray, shape: Union[int, Tuple]
) -> jax.numpy.ndarray:
    return jax.numpy.broadcast_to(arr=array, shape=shape)


def cast(a: jax.numpy.ndarray, dtype=None, casting="unsafe", copy=None):
    return a.astype(dtype=dtype)


def is_floating(a: jax.numpy.ndarray) -> bool:
    return jax.numpy.issubdtype(a.dtype, jax.numpy.floating)


def is_floating_dtype(dtype) -> bool:
    return is_floating(jax.numpy.empty((), dtype=dtype))


def to_numpy(*arrays: jax.numpy.ndarray) -> Tuple[np.ndarray, ...]:
    if len(arrays) == 1:
        return np.array(arrays[0])

    return tuple(np.array(arr) for arr in arrays)


def jit(f, *args, **kwargs):
    return jax.jit(f, *args, **kwargs)


def jit_method(f, *args, static_argnums=None, **kwargs):
    _static_argnums = (0,)

    if static_argnums is not None:
        _static_argnums += tuple(argnum + 1 for argnum in static_argnums)

    return jax.jit(f, *args, static_argnums=_static_argnums, **kwargs)
