"""Mean functions of random processes."""

import numpy as np

from .. import _function

__all__ = ["Zero"]


class Zero(_function.Function):
    """Zero mean function."""

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
            x,
            shape=x.shape[: x.ndim - self._input_ndim] + self._output_shape,
        )
