"""Logic functions on torch tensors."""
try:
    from torch import (  # pylint: disable=unused-import
        equal,
        greater,
        greater_equal,
        less,
        less_equal,
        logical_and,
        logical_not,
        logical_or,
        logical_xor,
        not_equal,
    )
except ModuleNotFoundError:
    pass

from typing import Optional, Union

from probnum.backend.typing import ShapeType


def all(
    a: "torch.Tensor",
    *,
    axis: Optional[Union[int, ShapeType]] = None,
    keepdims: bool = False
) -> "torch.Tensor":
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


def any(
    a: "torch.Tensor",
    *,
    axis: Optional[Union[int, ShapeType]] = None,
    keepdims: bool = False
) -> "torch.Tensor":
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
