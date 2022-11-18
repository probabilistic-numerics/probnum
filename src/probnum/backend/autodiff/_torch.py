"""(Automatic) Differentiation in PyTorch."""

from typing import Callable, Sequence, Union

try:
    import functorch
except ModuleNotFoundError:
    pass


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


def value_and_grad(
    fun: Callable, argnums: Union[int, Sequence[int]] = 0, has_aux: bool = False
) -> Callable:
    gfun_fun = functorch.grad_and_value(fun, argnums, has_aux=has_aux)

    def fun_gradfun(x):
        g, f = gfun_fun(x)
        return f, g

    return fun_gradfun
