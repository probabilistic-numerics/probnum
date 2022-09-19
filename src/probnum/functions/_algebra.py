r"""Algebraic operations on :class:`Function`\ s."""

from ._algebra_fallbacks import SumFunction
from ._function import Function
from ._zero import Zero

############
# Function #
############


@Function.__add__.register  # pylint: disable=no-member
def _(self, other: Function) -> SumFunction:
    return SumFunction(self, other)


@Function.__add__.register  # pylint: disable=no-member
def _(self, other: SumFunction) -> SumFunction:
    return SumFunction(self, *other.summands)


@Function.__add__.register  # pylint: disable=no-member
def _(self, other: Zero) -> Function:  # pylint: disable=unused-argument
    return self


@Function.__sub__.register  # pylint: disable=no-member
def _(self, other: Function) -> SumFunction:
    return SumFunction(self, -other)


@Function.__sub__.register  # pylint: disable=no-member
def _(self, other: Zero) -> Function:  # pylint: disable=unused-argument
    return self


###############
# SumFunction #
###############


@SumFunction.__add__.register  # pylint: disable=no-member
def _(self, other: Function) -> SumFunction:
    return SumFunction(*self.summands, other)


@SumFunction.__add__.register  # pylint: disable=no-member
def _(self, other: SumFunction) -> SumFunction:
    return SumFunction(*self.summands, *other.summands)


@SumFunction.__sub__.register  # pylint: disable=no-member
def _(self, other: Function) -> SumFunction:
    return SumFunction(*self.summands, -other)


########
# Zero #
########


@Zero.__add__.register  # pylint: disable=no-member
def _(self, other: Function) -> Function:  # pylint: disable=unused-argument
    return other


@Zero.__sub__.register  # pylint: disable=no-member
def _(self, other: Function) -> Function:  # pylint: disable=unused-argument
    return -other
