"""Utility functions to process function arguments."""

import numbers

import numpy as np
import scipy._lib._util

from probnum.types import (
    DTypeArgType,
    RandomStateArgType,
    RandomStateType,
    ScalarArgType,
    ShapeArgType,
    ShapeType,
)

__all__ = ["as_shape", "as_random_state", "as_numpy_scalar"]


def as_random_state(seed: RandomStateArgType) -> RandomStateType:
    """
    Transform a variable or RandomStateArgType into
    the random state format that is used internally.
    """
    return scipy._lib._util.check_random_state(seed=seed)


def as_shape(shape: ShapeArgType) -> ShapeType:
    """Transform a variable of ShapeArgType into a ShapeType (which is used internally)."""
    if isinstance(shape, (int, numbers.Integral, np.integer)):
        return (int(shape),)
    elif isinstance(shape, tuple) and all(isinstance(item, int) for item in shape):
        return shape
    else:
        try:
            _ = iter(shape)
        except TypeError as err:
            raise TypeError(
                f"The given shape {shape} must be an integer or an iterable of integers."
            ) from err

        if not all(isinstance(item, (int, numbers.Integral, np.integer)) for item in shape):
            raise TypeError(f"The given shape {shape} must only contain integer values.")

        return tuple(int(item) for item in shape)


def as_numpy_scalar(scalar: ScalarArgType, dtype: DTypeArgType = None) -> np.generic:
    """Transform a variable of ScalarArgType into a NumPy scalar (which is preferred internally)."""
    is_scalar = np.isscalar(scalar)
    is_scalar_array = isinstance(scalar, np.ndarray) and scalar.ndim == 0

    if not (is_scalar or is_scalar_array):
        raise ValueError("The given input is not a scalar.")

    return np.asarray(scalar, dtype=dtype)[()]
