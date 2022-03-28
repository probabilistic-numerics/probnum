from typing import Tuple, Union

import numpy as np

from probnum import backend, linops, randvars

__all__ = [
    "to_numpy",
    "cast",
]


def to_numpy(*xs: Union[backend.Array, linops.LinearOperator]) -> Tuple[np.ndarray]:
    res = []

    for x in xs:
        if backend.isarray(x):
            x = backend.to_numpy(x)
        elif isinstance(x, linops.LinearOperator):
            x = backend.to_numpy(x.todense())
        else:
            x = np.asarray(x)

        res.append(x)

    if len(xs) == 1:
        return res[0]

    return tuple(res)


def cast(a, dtype=None, casting="unsafe", copy=None):
    if isinstance(a, linops.LinearOperator):
        return a.astype(dtype=dtype, casting=casting, copy=copy)

    return backend.cast(a, dtype=dtype, casting=casting, copy=copy)


def atleast_1d(
    *objs: Union[
        backend.Array,
        linops.LinearOperator,
        randvars.RandomVariable,
    ]
) -> Union[
    Union[
        backend.Array,
        linops.LinearOperator,
        randvars.RandomVariable,
    ],
    Tuple[
        Union[
            backend.Array,
            linops.LinearOperator,
            randvars.RandomVariable,
        ],
        ...,
    ],
]:
    """Reshape arrays, linear operators and random variables to have at least 1
    dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    objs:
        One or more input linear operators, random variables or arrays.

    Returns
    -------
    res :
        An array / random variable / linop or tuple of arrays / random variables /
        linear operators, each with ``a.ndim >= 1``.
    """
    res = []

    for obj in objs:
        if isinstance(obj, np.ndarray):
            obj = np.atleast_1d(obj)
        elif isinstance(obj, backend.Array):
            obj = backend.atleast_1d(obj)
        elif isinstance(obj, randvars.RandomVariable):
            if obj.ndim == 0:
                obj = obj.reshape((1,))

        res.append(obj)

    if len(res) == 1:
        return res[0]

    return tuple(res)
