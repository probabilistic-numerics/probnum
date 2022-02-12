import abc
from typing import Callable

import numpy as np

from . import utils
from .typing import ArrayLike, ShapeLike, ShapeType


class Function(abc.ABC):
    def __init__(self, input_shape: ShapeLike, output_shape: ShapeLike) -> None:
        self._input_shape = utils.as_shape(input_shape)
        self._input_ndim = len(self._input_shape)

        self._output_shape = utils.as_shape(output_shape)
        self._output_ndim = len(self._output_shape)

    @property
    def input_shape(self) -> ShapeType:
        return self._input_shape

    @property
    def output_shape(self) -> ShapeType:
        return self._output_shape

    def __call__(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x)

        try:
            np.broadcast_to(
                x,
                shape=x.shape[: x.ndim - self._input_ndim] + self._input_shape,
            )
        except ValueError as ve:
            raise ValueError(
                f"The shape of the input {x.shape} is not compatible with the "
                f"specified `input_shape` of the `Function` {self._input_shape}."
            ) from ve

        res = self._evaluate(x)

        assert res.shape == (x.shape[: x.ndim - self._input_ndim] + self._output_shape)

        return res

    @abc.abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        pass


class LambdaFunction(Function):
    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        input_shape: ShapeLike,
        output_shape: ShapeLike,
    ) -> None:
        self._fn = fn

        super().__init__(input_shape, output_shape)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._fn(x)
