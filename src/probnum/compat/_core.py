import numpy as np

from probnum import backend, linops

__all__ = [
    "to_numpy",
    "cast",
]


def to_numpy(x):
    if isinstance(x, backend.ndarray):
        return backend.to_numpy(x)

    if isinstance(x, linops.LinearOperator):
        return backend.to_numpy(x.todense())

    return np.asarray(x)


def cast(a, dtype=None, casting="unsafe", copy=None):
    if isinstance(a, linops.LinearOperator):
        return a.astype(dtype=dtype, casting=casting, copy=copy)

    return backend.cast(a, dtype=dtype, casting=casting, copy=copy)
