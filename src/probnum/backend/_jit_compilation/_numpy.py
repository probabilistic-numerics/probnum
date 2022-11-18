"""Just-In-Time Compilation in NumPy."""

from typing import Callable, Iterable, Union


def jit(
    fun: Callable,
    *,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
):
    return fun


def jit_method(
    method: Callable,
    *,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
):
    return method
