"""Searching functions on NumPy arrays."""
from typing import Optional

import numpy as np
from numpy import nonzero, where  # pylint: disable=redefined-builtin, unused-import


def argmax(
    x: np.ndarray, /, *, axis: Optional[int] = None, keepdims: bool = False
) -> np.ndarray:
    return np.argmax(a=x, axis=axis, keepdims=keepdims)


def argmin(
    x: np.ndarray, /, *, axis: Optional[int] = None, keepdims: bool = False
) -> np.ndarray:
    return np.argmin(a=x, axis=axis, keepdims=keepdims)
