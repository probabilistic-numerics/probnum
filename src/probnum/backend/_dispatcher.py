from types import MethodType
from typing import Callable, Optional

from .._select_backend import BACKEND, Backend


class Dispatcher:
    """Dispatcher for backend-specific implementations of a function.

    Defines a decorator which can be used to define a function in multiple ways
    depending on the backend. This is useful, if besides the generic backend
    implementation, a more efficient implementation can be defined using
    functionality from a computation backend directly.

    Parameters
    ----------
    generic_impl
        Generic implementation.
    numpy_impl
        NumPy implementation.
    jax_impl
        JAX implementation.
    torch_impl
        PyTorch implementation.

    Example
    -------
    >>> @backend.Dispatcher
    ... def f(x):
    ...     raise NotImplementedError()
    ...
    ... @f.jax_impl
    ... def _(x: jnp.ndarray) -> jnp.ndarray:
    ...     pass
    """

    def __init__(
        self,
        generic_impl: Optional[Callable] = None,
        /,
        *,
        numpy_impl: Optional[Callable] = None,
        jax_impl: Optional[Callable] = None,
        torch_impl: Optional[Callable] = None,
    ):
        if generic_impl is None:
            generic_impl = Dispatcher._raise_not_implemented_error

        self._impl = {
            Backend.NUMPY: generic_impl if numpy_impl is None else numpy_impl,
            Backend.JAX: generic_impl if jax_impl is None else jax_impl,
            Backend.TORCH: generic_impl if torch_impl is None else torch_impl,
        }

    def numpy_impl(self, impl: Callable) -> Callable:
        self._impl[Backend.NUMPY] = impl

        return impl

    def jax_impl(self, impl: Callable) -> Callable:
        self._impl[Backend.JAX] = impl

        return impl

    def torch_impl(self, impl: Callable) -> Callable:
        self._impl[Backend.TORCH] = impl

        return impl

    def __call__(self, *args, **kwargs):
        return self._impl[BACKEND](*args, **kwargs)

    @staticmethod
    def _raise_not_implemented_error() -> None:
        raise NotImplementedError(
            f"This function is not implemented for the backend `{BACKEND.name}`"
        )

    def __get__(self, obj, objtype=None):
        """This is necessary in order to use the :class:`Dispatcher` as a class
        attribute which is then translated into a method of class instances, i.e. to
        allow for.

        .. code::

            class Foo:
                @Dispatcher
                def baz(self, x):
                    raise NotImplementedError()

                @baz.jax
                def _(self, x):
                    return x

            bar = Foo()
            bar.baz("Test")  # Output: "Test"

        See https://docs.python.org/3/howto/descriptor.html?highlight=methodtype#functions-and-methods
        for details.
        """
        if obj is None:
            return self

        return MethodType(self, obj)
