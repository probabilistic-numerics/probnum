import torch
from torch import (  # pylint: disable=redefined-builtin, unused-import, no-name-in-module
    Tensor as ndarray,
    as_tensor as asarray,
    atleast_1d,
    atleast_2d,
    bool,
    broadcast_shapes,
    broadcast_tensors as broadcast_arrays,
    cdouble,
    complex64 as csingle,
    diag,
    double,
    exp,
    eye,
    finfo,
    float as single,
    int32,
    int64,
    is_floating_point as is_floating,
    isfinite,
    linspace,
    log,
    maximum,
    pi,
    promote_types,
    sin,
    sqrt,
)

torch.set_default_dtype(torch.double)


def all(a: torch.Tensor, *, axis=None, keepdims: bool = False) -> torch.Tensor:
    if isinstance(axis, int):
        return torch.all(
            a,
            dim=axis,
            keepdim=keepdims,
        )

    axes = sorted(axis)

    res = a

    # If `keepdims is True`, this only works because axes is sorted!
    for axis in reversed(axes):
        res = torch.all(res, dim=axis, keepdims=keepdims)

    return res


def array(object, dtype=None, *, copy=True):
    if copy:
        return torch.tensor(object, dtype=dtype)

    return asarray(object, dtype=dtype)


def ndim(a):
    try:
        return a.ndim
    except AttributeError:
        return torch.as_tensor(a).ndim


def ones(shape, dtype=None):
    return torch.ones(shape, dtype=dtype)


def ones_like(a, dtype=None, *, shape=None):
    if shape is None:
        return torch.ones_like(input=a, dtype=dtype)

    return torch.ones(
        shape,
        dtype=a.dtype if dtype is None else dtype,
        layout=a.layout,
        device=a.device,
    )


def sum(a, axis=None, dtype=None, keepdims=False):
    if axis is None:
        axis = tuple(range(a.ndim))

    return torch.sum(a, dim=axis, keepdim=keepdims, dtype=dtype)


def zeros(shape, dtype=None):
    return torch.zeros(shape, dtype=dtype)


def zeros_like(a, dtype=None, *, shape=None):
    if shape is None:
        return torch.zeros_like(input=a, dtype=dtype)

    return torch.zeros(
        shape,
        dtype=a.dtype if dtype is None else dtype,
        layout=a.layout,
        device=a.device,
    )


def cast(a: torch.Tensor, dtype=None, casting="unsafe", copy=None):
    return a.to(dtype=dtype, copy=copy)


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f


inf = float("inf")
