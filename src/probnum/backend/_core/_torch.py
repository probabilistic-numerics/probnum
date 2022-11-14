from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import (  # pylint: disable=redefined-builtin, unused-import, no-name-in-module
    abs,
    atleast_1d,
    atleast_2d,
    broadcast_shapes,
    broadcast_tensors as broadcast_arrays,
    diag,
    diagonal,
    einsum,
    exp,
    eye,
    finfo,
    hstack,
    is_floating_point as is_floating,
    isfinite,
    kron,
    linspace,
    log,
    max,
    maximum,
    minimum,
    moveaxis,
    promote_types,
    reshape,
    result_type,
    sign,
    sin,
    sqrt,
    squeeze,
    stack,
    swapaxes,
    vstack,
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


def any(a: torch.Tensor, *, axis=None, keepdims: bool = False) -> torch.Tensor:
    if axis is None:
        return torch.any(a)

    if isinstance(axis, int):
        return torch.any(
            a,
            dim=axis,
            keepdim=keepdims,
        )

    axes = sorted(axis)

    res = a

    # If `keepdims is True`, this only works because axes is sorted!
    for axis in reversed(axes):
        res = torch.any(res, dim=axis, keepdims=keepdims)

    return res


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


def zeros_like(a, dtype=None, *, shape=None):
    if shape is None:
        return torch.zeros_like(input=a, dtype=dtype)

    return torch.zeros(
        shape,
        dtype=a.dtype if dtype is None else dtype,
        layout=a.layout,
        device=a.device,
    )


def to_numpy(*arrays: torch.Tensor) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if len(arrays) == 1:
        return arrays[0].cpu().detach().numpy()

    return tuple(arr.cpu().detach().numpy() for arr in arrays)


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f


def vectorize(pyfunc, /, *, excluded=None, signature=None):
    raise NotImplementedError()
