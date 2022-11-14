"""(Automatic) Differentiation in NumPy."""

from typing import Any, Callable, Sequence, Union


def grad(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, has_aux: bool = False
) -> Callable:
    raise NotImplementedError()


def hessian(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, has_aux: bool = False
) -> Callable:
    raise NotImplementedError


def vmap(
    fun: Callable,
    in_axes: Union[int, Sequence[Any]] = 0,
    out_axes: Union[int, Sequence[Any]] = 0,
) -> Callable:
    raise NotImplementedError


def jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    raise NotImplementedError


def jacfwd(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    raise NotImplementedError
