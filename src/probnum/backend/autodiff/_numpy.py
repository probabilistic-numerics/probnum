"""Differentiation in NumPy."""

from typing import Callable, Sequence, Union


def grad(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, has_aux: bool = False
) -> Callable:
    raise NotImplementedError()
