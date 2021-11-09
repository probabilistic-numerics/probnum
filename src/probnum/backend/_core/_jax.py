import jax
from jax.numpy import (  # pylint: disable=redefined-builtin, unused-import
    all,
    array,
    asarray,
    atleast_1d,
    atleast_2d,
    bool_ as bool,
    broadcast_arrays,
    broadcast_shapes,
    cdouble,
    complex64 as csingle,
    diag,
    double,
    exp,
    eye,
    finfo,
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

jax.config.update("jax_enable_x64", True)


def cast(a: jax.numpy.ndarray, dtype=None, casting="unsafe", copy=None):
    return a.astype(dtype=None)


def is_floating(a: jax.numpy.ndarray):
    return jax.numpy.issubdtype(a.dtype, jax.numpy.floating)


def jit(f, *args, **kwargs):
    return jax.jit(f, *args, **kwargs)


def jit_method(f, *args, static_argnums=None, **kwargs):
    _static_argnums = (0,)

    if static_argnums is not None:
        _static_argnums += tuple(argnum + 1 for argnum in static_argnums)

    return jax.jit(f, *args, static_argnums=_static_argnums, **kwargs)
