"""NumPy array creation functions."""
from typing import List, Optional, Union

import numpy as np
from numpy import diag, tril, triu  # pylint: disable= unused-import

from .. import Device, DType
from ..typing import ShapeType

# pylint: disable=redefined-builtin


def asarray(
    obj: Union[
        np.ndarray, bool, int, float, "NestedSequence", "SupportsBufferProtocol"
    ],
    /,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    copy: Optional[bool] = None,
) -> np.ndarray:
    if copy is None:
        copy = False
    return np.array(obj, dtype=dtype, copy=copy)


def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.arange(start, stop, step, dtype=dtype)


def empty(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


def empty_like(
    x: np.ndarray,
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.empty_like(x, shape=shape, dtype=dtype)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.eye(n_rows, n_cols, k=k, dtype=dtype)


def full(
    shape: ShapeType,
    fill_value: Union[int, float],
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.full(shape, fill_value, dtype=dtype)


def full_like(
    x: np.ndarray,
    /,
    fill_value: Union[int, float],
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.full_like(x, fill_value=fill_value, shape=shape, dtype=dtype)


def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> np.ndarray:
    return np.linspace(start, stop, num=num, dtype=dtype, endpoint=endpoint)


def meshgrid(*arrays: np.ndarray, indexing: str = "xy") -> List[np.ndarray]:
    return np.ones_like(*arrays, indexing=indexing)


def ones(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.ones(shape, dtype=dtype)


def ones_like(
    x: np.ndarray,
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.ones_like(x, shape=shape, dtype=dtype)


def zeros(
    shape: ShapeType,
    *,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def zeros_like(
    x: np.ndarray,
    /,
    *,
    shape: Optional[ShapeType] = None,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> np.ndarray:
    return np.zeros_like(x, shape=shape, dtype=dtype)
