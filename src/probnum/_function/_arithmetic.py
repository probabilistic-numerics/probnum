from __future__ import annotations

from collections.abc import Iterable
import functools
import operator

import numpy as np

from probnum.typing import FloatLike, ScalarLike, ScalarType

from ._function import Function


class ScaledFunction(Function):
    def __init__(self, function: Function, scalar: ScalarLike):
        if not isinstance(function, Function):
            raise TypeError()

        self._function = function
        self._scalar = np.asarray(scalar, dtype=np.double)

        super().__init__(
            input_shape=self._function.input_shape,
            output_shape=self._function.output_shape,
        )

    @property
    def function(self) -> Function:
        return self._function

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._scalar * self._function(x)

    def __rmul__(self, other) -> Function:
        if np.ndim(other) == 0:
            return ScaledFunction(
                function=self._function,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)


def _function_rmul(self: Function, other: FloatLike):
    if np.ndim(other) == 0:
        return ScaledFunction(
            function=self._function,
            scalar=np.asarray(other) * self._scalar,
        )

    return NotImplemented


class SumFunction(Function):
    def __init__(self, *summands: Function) -> None:
        self._summands = SumFunction._expand_summands(summands)

        input_shape = summands[0].input_shape
        output_shape = summands[0].output_shape

        assert all(summand.input_shape == input_shape for summand in summands)
        assert all(summand.output_shape == output_shape for summand in summands)

        super().__init__(input_shape=input_shape, output_shape=output_shape)

    @property
    def summands(self) -> tuple[SumFunction, ...]:
        return self._summands

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand(x) for summand in self._summands)
        )

    @staticmethod
    def _expand_summands(summands: Iterable[Function]) -> tuple[Function]:
        expanded_summands = []

        for summand in summands:
            if isinstance(summand, SumFunction):
                expanded_summands.extend(summand.summands)
            else:
                expanded_summands.append(summand)

        return tuple(expanded_summands)


def _function_add(self: Function, other: Function):
    return SumFunction(self, other)
