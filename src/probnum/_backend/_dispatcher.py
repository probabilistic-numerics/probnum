from typing import Any, Callable, Optional

from . import BACKEND, Backend


class BackendDispatcher:
    def __init__(
        self,
        numpy_impl: Optional[Callable[..., Any]],
        jax_impl: Optional[Callable[..., Any]] = None,
        pytorch_impl: Optional[Callable[..., Any]] = None,
    ):
        self._impl = {}

        if numpy_impl is not None:
            self._impl[Backend.NUMPY] = numpy_impl

        if jax_impl is not None:
            self._impl[Backend.JAX] = jax_impl

        if pytorch_impl is not None:
            self._impl[Backend.PYTORCH] = pytorch_impl

    def __call__(self, *args, **kwargs) -> Any:
        return self._impl[BACKEND](*args, **kwargs)
