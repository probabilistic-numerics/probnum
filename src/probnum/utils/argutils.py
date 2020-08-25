import numbers

import numpy as np
import scipy._lib._util

from probnum.typing import ShapeType, RandomStateType
from probnum._lib.argtypes import ShapeArgType, RandomStateArgType


def as_shape(x: ShapeArgType) -> ShapeType:
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


def as_random_state(x: RandomStateArgType) -> RandomStateType:
    return scipy._lib._util.check_random_state(x)
