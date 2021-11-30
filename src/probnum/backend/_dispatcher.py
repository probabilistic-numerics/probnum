from types import MethodType
from typing import Callable, Optional

from . import BACKEND, Backend


class Dispatcher:
    def __init__(
        self,
        numpy_impl: Optional[Callable] = None,
        jax_impl: Optional[Callable] = None,
        torch_impl: Optional[Callable] = None,
    ):
        self._impl = {}

        if numpy_impl is not None:
            self._impl[Backend.NUMPY] = numpy_impl

        if jax_impl is not None:
            self._impl[Backend.JAX] = jax_impl

        if torch_impl is not None:
            self._impl[Backend.TORCH] = torch_impl

    def numpy(self, impl: Callable) -> Callable:
        if Backend.NUMPY in self._impl:
            raise Exception()  # TODO

        self._impl[Backend.NUMPY] = impl

        return impl

    def jax(self, impl: Callable) -> Callable:
        if Backend.JAX in self._impl:
            raise Exception()  # TODO

        self._impl[Backend.JAX] = impl

        return impl

    def torch(self, impl: Callable) -> Callable:
        if Backend.TORCH in self._impl:
            raise Exception()  # TODO

        self._impl[Backend.TORCH] = impl

        return impl

    def __call__(self, *args, **kwargs):
        if BACKEND not in self._impl:
            raise NotImplementedError(
                f"This function is not implemented for the backend `{BACKEND.name}`"
            )
        return self._impl[BACKEND](*args, **kwargs)

    def __get__(self, obj, objtype=None):
        """This is necessary in order to use the :class:`Dispatcher` as a class
        attribute which is then translated into a method of class instances, i.e. to
        allow for

        .. code::

            class Foo:
                baz = Dispatcher()

                @bax.jax
                def _baz_jax(self, x):
                    return x

            bar = Foo()
            bar.baz("Test")  # Output: "Test"

        See https://docs.python.org/3/howto/descriptor.html?highlight=methodtype#functions-and-methods
        for details.
        """
        if obj is None:
            return self

        return MethodType(self, obj)
