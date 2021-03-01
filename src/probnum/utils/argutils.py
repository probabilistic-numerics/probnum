"""Utility functions for argument types."""

import numbers

import numpy as np
import scipy._lib._util

from probnum.type import (
    DTypeArgType,
    RandomStateArgType,
    RandomStateType,
    ScalarArgType,
    ShapeArgType,
    ShapeType,
)

__all__ = ["as_shape", "as_random_state", "as_numpy_scalar"]


def as_random_state(seed: RandomStateArgType) -> RandomStateType:
    """Turn ``seed`` into a np.random.RandomState instance.

    Parameters
    ----------
    seed
        If seed is None, return the RandomState singleton used by np.random. If seed is
        an int, return a new RandomState instance seeded with seed. If seed is already a
        RandomState instance, return it.

    Raises
    -------
    ValueError
        If seed is neither None, an int or a RandomState instance.
    """
    return scipy._lib._util.check_random_state(seed)


def as_shape(x: ShapeArgType) -> ShapeType:
    """Convert a shape representation into a shape defined as a tuple of ints.

    Parameters
    ----------
    x
        Shape representation.
    """
    if isinstance(x, (int, numbers.Integral, np.integer)):
        return (int(x),)
    elif isinstance(x, tuple) and all(isinstance(item, int) for item in x):
        return x
    else:
        try:
            _ = iter(x)
        except TypeError as e:
            raise TypeError(
                f"The given shape {x} must be an integer or an iterable of integers."
            ) from e

        if not all(isinstance(item, (int, numbers.Integral, np.integer)) for item in x):
            raise TypeError(f"The given shape {x} must only contain integer values.")

        return tuple(int(item) for item in x)


def as_numpy_scalar(x: ScalarArgType, dtype: DTypeArgType = None) -> np.generic:
    """Convert a scalar into a NumPy scalar.

    Parameters
    ----------
    x
        Scalar value.
    dtype
        Data type of the scalar.
    """

    if np.ndim(x) != 0:
        raise ValueError("The given input is not a scalar.")

    return np.asarray(x, dtype=dtype)[()]
