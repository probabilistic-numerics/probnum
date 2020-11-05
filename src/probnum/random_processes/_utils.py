"""Utility functions for random processes."""

from typing import Any

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

    """
    raise NotImplementedError
