"""Vectorization in NumPy."""

from typing import Any, Callable, Sequence, Union

from numpy import vectorize  # pylint: disable=redefined-builtin, unused-import


def vmap(
    fun: Callable,
    in_axes: Union[int, Sequence[Any]] = 0,
    out_axes: Union[int, Sequence[Any]] = 0,
) -> Callable:
    raise NotImplementedError
