"""(Automatic) Differentiation."""

from typing import Callable, Sequence, Union

from probnum import backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from . import _numpy as _impl
elif _backend.BACKEND is _backend.Backend.JAX:
    from . import _jax as _impl
elif _backend.BACKEND is _backend.Backend.TORCH:
    from . import _torch as _impl


def grad(fun: Callable, argnums: Union[int, Sequence[int]] = 0) -> Callable:
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
    >>> from probnum import backend
    >>> from probnum.backend.autodiff import grad
    >>> grad_sin = grad(backend.sin)
    >>> grad_sin(backend.pi)
    -1.0
    """
    return _impl.grad(fun=fun, argnums=argnums)
