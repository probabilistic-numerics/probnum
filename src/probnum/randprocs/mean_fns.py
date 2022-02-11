import numpy as np

from .. import _function
from ..typing import ShapeLike

__all__ = ["Zero"]


class MeanFunction(_function.Function):
    def __init__(self, input_shape: ShapeLike, output_shape: ShapeLike = ()):
        super().__init__(input_shape, output_shape)


class Zero(MeanFunction):
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x, shape=x.shape[:-1] + self._output_shape)
