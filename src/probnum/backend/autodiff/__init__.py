"""(Automatic) Differentiation."""

from typing import Any, Callable, Sequence, Tuple, Union

from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _impl
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _impl
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _impl


__all__ = [
    "grad",
    "hessian",
    "jacfwd",
    "jacrev",
    "value_and_grad",
]
__all__.sort()


def grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    """Creates a function that evaluates the gradient of ``fun``.

    Parameters
    ----------
    fun
        Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, or standard Python containers.
        Argument arrays in the positions specified by ``argnums`` must be of
        inexact (i.e., floating-point or complex) type. It
        should return a scalar (which includes arrays with shape ``()`` but not
        arrays with shape ``(1,)`` etc.)
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the first element is considered
        the output of the mathematical function to be differentiated and the second
        element is auxiliary data.

    Returns
    -------
    grad_fun
        A function with the same arguments as ``fun``, that evaluates the gradient
        of ``fun``. If ``argnums`` is an integer then the gradient has the same
        shape and type as the positional argument indicated by that integer. If
        argnums is a tuple of integers, the gradient is a tuple of values with the
        same shapes and types as the corresponding arguments.

    Examples
    --------
    >>> from probnum.backend.autodiff import grad
    >>> grad_sin = grad(backend.sin)
    >>> grad_sin(backend.pi)
    -1.0
    """
    return _impl.grad(fun=fun, argnums=argnums, has_aux=has_aux)


def hessian(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    """Hessian of ``fun`` as a dense array.

    Parameters
    ----------
    fun
        Function whose Hessian is to be computed.  Its arguments at positions
        specified by ``argnums`` should be arrays, scalars, or standard Python
        containers thereof. It should return arrays, scalars, or standard Python
        containers thereof.
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.

    Returns
    -------
    hessian
        A function with the same arguments as ``fun``, that evaluates the Hessian of
        ``fun``.

    >>> from probnum.backend.autodiff import hessian
    >>> g = lambda x: x[0]**3 - 2*x[0]*x[1] - x[1]**6
    >>> hessian(g)(backend.asarray([1., 2.])))
    [[   6.   -2.]
    [  -2. -480.]]
    """
    return _impl.hessian(fun=fun, argnums=argnums, has_aux=has_aux)


def jacfwd(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

    Parameters
    ----------
    fun
        Function whose Jacobian is to be computed.
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.

    Returns
    -------
    jacfun
      A function with the same arguments as ``fun``, that evaluates the Jacobian of
      ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
      then a pair of (jacobian, auxiliary_data) is returned.
    """
    return _impl.jacfwd(fun, argnums, has_aux=has_aux)


def jacrev(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable:
    """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    Parameters
    ----------
    fun
        Function whose Jacobian is to be computed.
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data.

    Returns
    -------
    jacfun
      A function with the same arguments as ``fun``, that evaluates the Jacobian of
      ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
      then a pair of (jacobian, auxiliary_data) is returned.
    """
    return _impl.jacrev(fun, argnums, has_aux=has_aux)


def value_and_grad(
    fun: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    *,
    has_aux: bool = False,
) -> Callable[..., Tuple[Any, Any]]:
    """Create a function that efficiently evaluates both ``fun`` and the gradient of
    ``fun``.

    Parameters
    ----------
    fun
        Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, or standard Python containers. It should
        return a scalar (which includes arrays with shape ``()`` but not arrays with
        shape ``(1,)`` etc.)
    argnums
        Specifies which positional argument(s) to differentiate with respect to.
    has_aux
        Indicates whether ``fun`` returns a pair where the first element is considered
        the output of the mathematical function to be differentiated and the second
        element is auxiliary data.

    Returns
    -------
    value_and_grad
        A function with the same arguments as ``fun`` that evaluates both ``fun`` and
        the gradient of ``fun`` and returns them as a pair (a two-element tuple). If
        ``argnums`` is an integer then the gradient has the same shape and type as the
        positional argument indicated by that integer. If ``argnums`` is a sequence of
        integers, the gradient is a tuple of values with the same shapes and types as
        the corresponding arguments. If ``has_aux`` is ``True`` then a tuple of
        ``((value, auxiliary_data), gradient)`` is returned.
    """
    return _impl.value_and_grad(fun, argnums, has_aux=has_aux)
