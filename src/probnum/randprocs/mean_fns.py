"""Mean functions of random processes."""

import numpy as np

from .. import _function
from ..typing import ShapeLike

__all__ = ["Zero"]


class MeanFunction(_function.Function):
    """Mean function of a random process."""

    def __init__(self, input_shape: ShapeLike, output_shape: ShapeLike = ()):
        super().__init__(input_shape, output_shape)


class Zero(MeanFunction):
    """Zero mean function."""

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
            x,
            shape=x.shape[: x.ndim - self._input_ndim] + self._output_shape,
        )
