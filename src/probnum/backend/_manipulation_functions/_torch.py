"""Torch tensor manipulation functions."""

from typing import List, Optional, Tuple, Union

try:
    import torch
    from torch import (  # pylint: disable=unused-import
        atleast_1d,
        atleast_2d,
        broadcast_shapes,
        broadcast_tensors as broadcast_arrays,
        hstack,
        movedim as move_axes,
        vstack,
    )
except ModuleNotFoundError:
    pass

from ..typing import ShapeType


def broadcast_to(x: "torch.Tensor", /, shape: ShapeType) -> "torch.Tensor":
    return torch.broadcast_to(x, size=shape)


def concat(
    arrays: Union[Tuple["torch.Tensor", ...], List["torch.Tensor"]],
    /,
    *,
    axis: Optional[int] = 0,
) -> "torch.Tensor":
    return torch.concat(tensors=arrays, dim=axis)


def expand_axes(x: "torch.Tensor", /, *, axis: int = 0) -> "torch.Tensor":
    return torch.unsqueeze(input=x, dim=axis)


def flip(
    x: "torch.Tensor", /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> "torch.Tensor":
    return torch.flip(x, dims=axis)


def permute_axes(x: "torch.Tensor", /, axes: Tuple[int, ...]) -> "torch.Tensor":
    return torch.permute(x, dims=axes)


def swap_axes(x: "torch.Tensor", /, axis1: int, axis2: int) -> "torch.Tensor":
    return torch.swapdims(x, dim0=axis1, dim1=axis2)


def reshape(
    x: "torch.Tensor", /, shape: ShapeType, *, copy: Optional[bool] = None
) -> "torch.Tensor":
    if copy is not None:
        if copy:
            out = torch.clone(x)
    return torch.reshape(out, shape=shape)


def roll(
    x: "torch.Tensor",
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> "torch.Tensor":
    return torch.roll(x, shifts=shift, dims=axis)


def squeeze(x: "torch.Tensor", /, axis: Union[int, Tuple[int, ...]]) -> "torch.Tensor":
    return torch.squeeze(x, dim=axis)


def stack(
    arrays: Union[Tuple["torch.Tensor", ...], List["torch.Tensor"]], /, *, axis: int = 0
) -> "torch.Tensor":
    return torch.stack(arrays, dim=axis)


def tile(A: "torch.Tensor", reps: "torch.Tensor") -> "torch.Tensor":
    return torch.tile(input=A, dims=reps)
