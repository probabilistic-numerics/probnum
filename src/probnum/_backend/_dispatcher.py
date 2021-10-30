from typing import Callable, Optional

from . import BACKEND, Backend


class Dispatcher:
    def __init__(
        self,
        numpy_impl: Optional[Callable] = None,
        jax_impl: Optional[Callable] = None,
        pytorch_impl: Optional[Callable] = None,
    ):
        self._impl = {}

        if numpy_impl is not None:
            self._impl[Backend.NUMPY] = numpy_impl

        if jax_impl is not None:
            self._impl[Backend.JAX] = jax_impl

        if pytorch_impl is not None:
            self._impl[Backend.PYTORCH] = pytorch_impl

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
        if Backend.PYTORCH in self._impl:
            raise Exception()  # TODO

        self._impl[Backend.PYTORCH] = impl

        return impl

    def __call__(self, *args, **kwargs):
        return self._impl[BACKEND](*args, **kwargs)
