"""Torch tensor creation functions."""
from typing import List, Optional, Union

try:
    import torch
    from torch import tril, triu  # pylint: unused-import
except ModuleNotFoundError:
    pass
from .. import Device, DType
from ... import config
from .._data_types import is_floating_dtype
from ..typing import ShapeType

# pylint: disable=redefined-builtin


def asarray(
    obj: Union[
        "torch.Tensor", bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
    copy: Optional[bool] = None,
) -> "torch.Tensor":
    out = torch.as_tensor(obj, dtype=dtype, device=device)

    if is_floating_dtype(out.dtype):
        out = out.to(dtype=config.default_floating_dtype, copy=False)

    if copy is None:
        copy = False
    if copy:
        return out.clone()

    return out


def diag(x: "torch.Tensor", /, *, k: int = 0) -> "torch.Tensor":
    return torch.diag(x, diagonal=k)


def tril(x: "torch.Tensor", /, k: int = 0) -> "torch.Tensor":
    return tril(x, diagonal=k)


def triu(x: "torch.Tensor", /, k: int = 0) -> "torch.Tensor":
    return triu(x, diagonal=k)


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.arange(start=start, stop=stop, step=step, dtype=dtype, device=device)


def empty(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.empty(shape, dtype=dtype, device=device)


def empty_like(
    x: "torch.Tensor",
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.empty_like(x, layout=shape, dtype=dtype, device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    if k != 0:
        raise NotImplementedError
    if n_cols is None:
        return torch.eye(n_rows, dtype=dtype, device=device)
    return torch.eye(n_rows, n_cols, dtype=dtype, device=device)


def full(
    shape: ShapeType,
    fill_value: Union[int, float],
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.full(shape, fill_value, dtype=dtype, device=device)


def full_like(
    x: "torch.Tensor",
    /,
    fill_value: Union[int, float],
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.full_like(
        x, fill_value=fill_value, layout=shape, dtype=dtype, device=device
    )


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> "torch.Tensor":
    if not endpoint:
        raise NotImplementedError

    return torch.linspace(start=start, end=stop, steps=num, dtype=dtype, device=device)


def meshgrid(*arrays: "torch.Tensor", indexing: str = "xy") -> List["torch.Tensor"]:
    return torch.meshgrid(*arrays, indexing=indexing)


def ones(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.ones(shape, dtype=dtype, device=device)


def ones_like(
    x: "torch.Tensor",
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.ones_like(x, layout=shape, dtype=dtype, device=device)


def zeros(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.zeros(shape, dtype=dtype, device=device)


def zeros_like(
    x: "torch.Tensor",
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> "torch.Tensor":
    return torch.zeros_like(x, layout=shape, dtype=dtype, device=device)
