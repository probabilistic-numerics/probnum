"""Just-In-Time Compilation."""
from typing import Callable, Iterable, Union

from ..._select_backend import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = ["jit", "jit_method"]


def jit(
    fun: Callable,
    *,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
):
    """Set up ``fun`` for just-in-time compilation.

    Parameters
    ----------
    fun
        Function to be jitted. ``fun`` should be a pure function, as side-effects may
        only be executed once. The arguments and return value of ``fun`` should be
        arrays, scalars, or (nested) standard Python containers (tuple/list/dict)
        thereof.
    static_argnums
        An optional int or collection of ints that specify which positional arguments to
        treat as static (compile-time constant). Operations that only depend on static
        arguments will be constant-folded in Python (during tracing), and so the
        corresponding argument values can be any Python object.
    static_argnames
        An optional string or collection of strings specifying which named arguments to
        treat as static (compile-time constant).

    Returns
    -------
    wrapped
        A wrapped version of ``fun``, set up for just-in-time compilation.
    """
    return _impl.jit(
        fun, static_argnums=static_argnums, static_argnames=static_argnames
    )


def jit_method(
    method: Callable,
    *,
    static_argnums: Union[int, Iterable[int], None] = None,
    static_argnames: Union[str, Iterable[str], None] = None,
):
    """Set up a ``method`` of an object for just-in-time compilation.

    Convencience wrapper for jitting the method(s) of an object. Typically used as a
    decorator.

    Parameters
    ----------
    method
        Method to be jitted. ``method`` should be a pure function, as side-effects may
        only be executed once. The arguments and return value of ``method`` should be
        arrays, scalars, or (nested) standard Python containers (tuple/list/dict)
        thereof.
    static_argnums
        An optional int or collection of ints that specify which positional arguments to
        treat as static (compile-time constant). Operations that only depend on static
        arguments will be constant-folded in Python (during tracing), and so the
        corresponding argument values can be any Python object.
    static_argnames
        An optional string or collection of strings specifying which named arguments to
        treat as static (compile-time constant).

    Returns
    -------
    wrapped
        A wrapped version of ``method``, set up for just-in-time compilation.
    """
    return _impl.jit_method(
        method, static_argnums=static_argnums, static_argnames=static_argnames
    )
