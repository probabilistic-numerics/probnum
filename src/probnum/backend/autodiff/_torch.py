"""(Automatic) Differentiation in PyTorch."""

from typing import Any, Callable, Sequence, Union

import functorch


def grad(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, has_aux: bool = False
) -> Callable:
    return functorch.grad(fun, argnums, has_aux=has_aux)


def hessian(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, has_aux: bool = False
) -> Callable:
    return functorch.jacfwd(
        functorch.jacrev(fun, argnums, has_aux=has_aux), argnums, has_aux=has_aux
    )


def vmap(
    fun: Callable,
    in_axes: Union[int, Sequence[Any]] = 0,
    out_axes: Union[int, Sequence[Any]] = 0,
) -> Callable:
    return functorch.vmap(fun, in_dims=in_axes, out_dims=out_axes)


def jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    return functorch.jacrev(fun, argnums, has_aux=has_aux)


def jacfwd(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    return functorch.jacfwd(fun, argnums, has_aux=has_aux)
