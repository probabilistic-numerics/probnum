from typing import Tuple, Union

import numpy as np

from probnum import backend, linops

__all__ = [
    "to_numpy",
    "cast",
]


def to_numpy(*xs: Union[backend.ndarray, linops.LinearOperator]) -> Tuple[np.ndarray]:
    res = []

    for x in xs:
        if isinstance(x, backend.ndarray):
            x = backend.to_numpy(x)
        elif isinstance(x, linops.LinearOperator):
            x = backend.to_numpy(x.todense())
        else:
            x = np.asarray(x)

        res.append(x)

    return tuple(res)


def cast(a, dtype=None, casting="unsafe", copy=None):
    if isinstance(a, linops.LinearOperator):
        return a.astype(dtype=dtype, casting=casting, copy=copy)

    return backend.cast(a, dtype=dtype, casting=casting, copy=copy)
