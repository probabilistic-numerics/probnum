"""Algebraic operations on functions."""

from ._algebra_fallbacks import SumFunction
from ._function import Function


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
