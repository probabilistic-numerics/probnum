"""Utility functions for random processes."""

from typing import Any

import numpy as np

import probnum

from . import _random_process


def asrandproc(obj: Any) -> _random_process.RandomProcess:
    """Convert ``obj`` to a :class:`RandomProcess`.

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
    >>> f = pn.asrandproc(lambda x : x ** 2 + 1.0)
    >>> f
    <RandomProcess with input_dim=1, output_dim=1, dtype=float64>
    >>> f(2)
    <Constant with shape=(), dtype=float64>
    >>> f.mean(2)
    5.0
    """
    # TODO replace this function with an initialization in Random Process
    if isinstance(obj, _random_process.RandomProcess):
        return obj
    elif callable(obj):
        return _random_process.RandomProcess(
            input_dim=1,
            output_dim=1,
            dtype=np.dtype(np.float_),
            fun=lambda x: probnum.asrandvar(obj(x)),
            mean=lambda x: probnum.asrandvar(obj(x)).mean,
            # cov=lambda x: probnum.asrandvar(obj(x)).cov,
            sample_at_input=lambda x, size: probnum.asrandvar(obj(x)).sample(size),
        )
    else:
        raise ValueError(
            f"Argument of type {type(obj)} cannot be converted to a random process."
        )
