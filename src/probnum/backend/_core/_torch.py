from typing import Sequence, Tuple, Union

import numpy as np
import torch
from torch import (  # pylint: disable=redefined-builtin, unused-import, no-name-in-module
    Tensor as ndarray,
    abs,
    as_tensor as asarray,
    atleast_1d,
    atleast_2d,
    bool,
    broadcast_shapes,
    broadcast_tensors as broadcast_arrays,
    cdouble,
    complex64 as csingle,
    diag,
    diagonal,
    double,
    dtype,
    einsum,
    exp,
    eye,
    finfo,
    float as single,
    hstack,
    int32,
    int64,
    is_floating_point as is_floating,
    isfinite,
    linspace,
    log,
    maximum,
    moveaxis,
    pi,
    promote_types,
    reshape,
    sign,
    sin,
    sqrt,
    squeeze,
    stack,
    swapaxes,
    vstack,
)

torch.set_default_dtype(torch.double)


def broadcast_to(array: torch.Tensor, shape: Union[int, Tuple]) -> torch.Tensor:
    return torch.broadcast_to(input=array, size=tuple(shape))


def asdtype(x) -> torch.dtype:
    if isinstance(x, torch.dtype):
        return x

    return torch.as_tensor(
        np.empty(
            (),
            dtype=np.dtype(x),
        ),
    ).dtype


def is_floating_dtype(dtype) -> bool:
    return is_floating(torch.empty((), dtype=dtype))


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


def full(
    shape,
    fill_value,
    dtype=None,
) -> torch.Tensor:
    return torch.full(
        size=shape,
        fill_value=fill_value,
        dtype=dtype,
    )


def full_like(
    a: torch.Tensor,
    fill_value,
    dtype=None,
    shape=None,
) -> torch.Tensor:
    return torch.full(
        shape if shape is not None else a.shape,
        fill_value,
        dtype=dtype if dtype is not None else a.dtype,
        layout=a.layout,
        device=a.device,
        requires_grad=a.requires_grad,
    )


def tile(A: torch.Tensor, reps: torch.Tensor) -> torch.Tensor:
    return torch.tile(input=A, dims=reps)


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


def concatenate(arrays: Sequence[torch.Tensor], axis: int = 0) -> torch.Tensor:
    return torch.cat(tensors=arrays, dim=axis)


def expand_dims(a: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.unsqueeze(input=a, dim=axis)


def cast(a: torch.Tensor, dtype=None, casting="unsafe", copy=None):
    return a.to(dtype=dtype, copy=copy)


def to_numpy(*arrays: torch.Tensor) -> Tuple[np.ndarray, ...]:
    if len(arrays) == 1:
        return arrays[0].cpu().detach().numpy()

    return tuple(arr.cpu().detach().numpy() for arr in arrays)


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f


inf = float("inf")