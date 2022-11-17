from typing import Tuple, Union

import numpy as np
import torch
from torch import (  # pylint: disable=redefined-builtin, unused-import, no-name-in-module
    abs,
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


def jit(f, *args, **kwargs):
    return f


def jit_method(f, *args, **kwargs):
    return f


def vectorize(pyfunc, /, *, excluded=None, signature=None):
    raise NotImplementedError()
