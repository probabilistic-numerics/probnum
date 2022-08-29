from __future__ import annotations

import functools
import operator

import numpy as np

from probnum.typing import ScalarLike, ScalarType

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

    @functools.singledispatchmethod
    def __rmul__(self, other):
        if np.ndim(other) == 0:
            return ScaledFunction(
                function=self._function,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)


class SumFunction(Function):
    def __init__(self, *summands: Function) -> None:
        self._summands = summands

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

    @functools.singledispatchmethod
    def __add__(self, other):
        return super().__add__(other)


@Function.__add__.register  # pylint: disable=no-member
def _(self, other: Function) -> SumFunction:
    return SumFunction(self, other)


@Function.__add__.register  # pylint: disable=no-member
def _(self, other: SumFunction) -> SumFunction:
    return SumFunction(self, *other.summands)


@Function.__sub__.register  # pylint: disable=no-member
def _(self, other: Function) -> SumFunction:
    return SumFunction(self, -other)


@SumFunction.__add__.register  # pylint: disable=no-member
def _(self, other: Function) -> SumFunction:
    return SumFunction(*self.summands, other)


@SumFunction.__add__.register  # pylint: disable=no-member
def _(self, other: SumFunction) -> SumFunction:
    return SumFunction(*self.summands, *other.summands)


@SumFunction.__sub__.register  # pylint: disable=no-member
def _(self, other: Function) -> SumFunction:
    return SumFunction(*self.summands, -other)
