from typing import Tuple, Union

import jax
from jax.numpy import (  # pylint: disable=redefined-builtin, unused-import
    abs,
    all,
    any,
    arange,
    broadcast_arrays,
    broadcast_shapes,
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
    reshape,
    sign,
    sin,
    sqrt,
    squeeze,
    stack,
    sum,
    swapaxes,
    vstack,
    zeros,
    zeros_like,
)
import numpy as np

jax.config.update("jax_enable_x64", True)


def to_numpy(*arrays: jax.numpy.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if len(arrays) == 1:
        return np.array(arrays[0])

    return tuple(np.array(arr) for arr in arrays)


def vectorize(pyfunc, /, *, excluded, signature):
    return jax.numpy.vectorize(
        pyfunc,
        excluded=excluded if excluded is not None else set(),
        signature=signature,
    )


def jit(f, *args, **kwargs):
    return jax.jit(f, *args, **kwargs)


def jit_method(f, *args, static_argnums=None, **kwargs):
    _static_argnums = (0,)

    if static_argnums is not None:
        _static_argnums += tuple(argnum + 1 for argnum in static_argnums)

    return jax.jit(f, *args, static_argnums=_static_argnums, **kwargs)
