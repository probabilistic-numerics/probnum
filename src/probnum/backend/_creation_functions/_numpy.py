"""NumPy array creation functions."""
from typing import List, Optional, Union

import numpy as np
from numpy import (  # pylint: disable=redefined-builtin, unused-import
    arange,
    empty,
    empty_like,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    tril,
    triu,
    zeros,
    zeros_like,
)

from .. import Array, Device, Dtype
from ..typing import ShapeType


def asarray(
    obj: Union[
        np.ndarray, bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> np.ndarray:
    if copy is None:
        copy = False
    return np.array(obj, dtype=dtype, copy=copy)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.eye(n_rows, n_cols, k=k, dtype=dtype)


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.arange(start, stop, step, dtype=dtype)


def empty(
    shape: ShapeType,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.empty(shape, dtype=dtype)


def empty_like(
    x: Array,
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.empty_like(x, dtype=dtype)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.eye(n_rows, n_cols, k=k, dtype=dtype)


def full(
    shape: ShapeType,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.full(shape, fill_value, dtype=dtype)


def full_like(
    x: Array,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.full_like(x, fill_value=fill_value, dtype=dtype)


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> Array:
    return np.linspace(start, stop, num=num, dtype=dtype, endpoint=endpoint)


def meshgrid(*arrays: Array, indexing: str = "xy") -> List[Array]:
    return np.ones_like(*arrays, indexing=indexing)


def ones(
    shape: ShapeType,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.ones(shape, dtype=dtype)


def ones_like(
    x: Array,
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.ones_like(x, dtype=dtype)


def zeros(
    shape: ShapeType,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.zeros(shape, dtype=dtype)


def zeros_like(
    x: Array,
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> Array:
    return np.zeros_like(x, dtype=dtype)
