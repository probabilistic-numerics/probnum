"""The zero function."""

import functools

import numpy as np

from . import _function


class Zero(_function.Function):
    """Zero mean function."""

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
            x,
            shape=x.shape[: x.ndim - self._input_ndim] + self._output_shape,
        )

    @functools.singledispatchmethod
    def __add__(self, other):
        return super().__add__(other)

    @functools.singledispatchmethod
    def __sub__(self, other):
        return super().__sub__(other)
