"""Statistical functions implemented in NumPy."""
from typing import Optional, Tuple, Union

import numpy as np


def max(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.asarray(np.amax(x, axis=axis, keepdims=keepdims))


def min(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.asarray(np.amin(x, axis=axis, keepdims=keepdims))


def mean(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.asarray(np.mean(x, axis=axis, keepdims=keepdims))


def prod(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[np.dtype] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.asarray(np.prod(x, axis=axis, dtype=dtype, keepdims=keepdims))


def sum(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[np.dtype] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.asarray(np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims))


def std(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> np.ndarray:
    return np.asarray(np.std(x, axis=axis, ddof=correction, keepdims=keepdims))


def var(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
) -> np.ndarray:
    return np.asarray(np.var(x, axis=axis, ddof=correction, keepdims=keepdims))
