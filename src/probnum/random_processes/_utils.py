"""Utility functions for random processes."""

from typing import Any

import numpy as np

from . import _random_process


def asrandproc(obj: Any) -> _random_process.RandomProcess:
    """
    Convert ``obj`` to a :class:`RandomProcess`.

    Converts an object such as functions or random process-type
    objects to a ProbNum :class:`RandomProcess`.

    Parameters
    ----------
    obj :
        Object to be represented as a :class:`RandomProcess`.

    See Also
    --------
    RandomProcess : Class representing random processes.

    Examples
    --------
    >>> import probnum as pn
    >>> f = lambda x : x ** 2 + 1.0
    >>> rp = pn.asrandproc(f)
    >>> rp(2)
    5.0
    >>> rp
    <RandomProcess with input_shape=(), output_shape=(), dtype=float64>
    """
    if isinstance(obj, _random_process.RandomProcess):
        return obj
    elif callable(obj):
        return _random_process.RandomProcess(
            input_shape=(), output_shape=(), dtype=np.dtype(np.float_), fun=obj
        )
    else:
        raise ValueError(
            f"Argument of type {type(obj)} cannot be converted to a random process."
        )
