import numpy as np

__all__ = ["as_numpy_scalar"]


def as_numpy_scalar(x, dtype=None):
    if not np.isscalar(x):
        raise ValueError("The given input is not a scalar")

    return np.asarray([x], dtype=dtype)[0]
