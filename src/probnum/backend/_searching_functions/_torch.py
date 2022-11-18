"""Searching functions on torch tensors."""
from typing import Optional, Tuple

try:
    import torch
    from torch import (  # pylint: disable=redefined-builtin, unused-import, no-name-in-module
        where,
    )
except ModuleNotFoundError:
    pass


def argmax(
    x: "torch.Tensor", /, *, axis: Optional[int] = None, keepdims: bool = False
) -> "torch.Tensor":
    return torch.argmax(input=x, dim=axis, keepdim=keepdims)


def argmin(
    x: "torch.Tensor", /, *, axis: Optional[int] = None, keepdims: bool = False
) -> "torch.Tensor":
    return torch.argmin(input=x, dim=axis, keepdim=keepdims)


def nonzero(x: "torch.Tensor", /) -> Tuple["torch.Tensor", ...]:
    return torch.nonzero(input=x, as_tuple=True)
