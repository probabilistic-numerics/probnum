"""The zero function."""

import functools

from probnum import backend

from . import _function


class Zero(_function.Function):
    """Zero mean function."""

    def _evaluate(self, x: backend.Array) -> backend.Array:
        return backend.zeros_like(
            x,
            shape=x.shape[: x.ndim - self._input_ndim] + self._output_shape,
        )

    @functools.singledispatchmethod
    def __add__(self, other):
        return super().__add__(other)

    @functools.singledispatchmethod
    def __sub__(self, other):
        return super().__sub__(other)
