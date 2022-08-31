r"""Fallback implementation for algebraic operations on :class:`Function`\ s."""

from __future__ import annotations

import functools
import operator

import numpy as np

from probnum import utils
from probnum.typing import ScalarLike, ScalarType

from ._function import Function


class SumFunction(Function):
    r"""Pointwise sum of :class:`Function`\ s.

    Given functions :math:`f_1, \dotsc, f_n \colon \mathbb{R}^n \to \mathbb{R}^m`, this
    defines a new function

    .. math::
        \sum_{i = 1}^n f_i \colon \mathbb{R}^n \to \mathbb{R}^m,
        x \mapsto \sum_{i = 1}^n f_i(x).

    Parameters
    ----------
    *summands
        The functions :math:`f_1, \dotsc, f_n`.
    """

    def __init__(self, *summands: Function) -> None:
        if not all(isinstance(summand, Function) for summand in summands):
            raise TypeError(
                "The functions to be added must be objects of type `Function`."
            )

        if not all(
            summand.input_shape == summands[0].input_shape for summand in summands
        ):
            raise ValueError(
                "The functions to be added must all have the same input shape."
            )

        if not all(
            summand.output_shape == summands[0].output_shape for summand in summands
        ):
            raise ValueError(
                "The functions to be added must all have the same output shape."
            )

        self._summands = summands

        super().__init__(
            input_shape=summands[0].input_shape,
            output_shape=summands[0].output_shape,
        )

    @property
    def summands(self) -> tuple[SumFunction, ...]:
        r"""The functions :math:`f_1, \dotsc, f_n` to be added."""
        return self._summands

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand(x) for summand in self._summands)
        )

    @functools.singledispatchmethod
    def __add__(self, other):
        return super().__add__(other)

    @functools.singledispatchmethod
    def __sub__(self, other):
        return super().__sub__(other)


class ScaledFunction(Function):
    r"""Function multiplied pointwise with a scalar.

    Given a function :math:`f \colon \mathbb{R}^n \to \mathbb{R}^m` and a scalar
    :math:`\alpha \in \mathbb{R}`, this defines a new function

    .. math::
        \alpha f \colon \mathbb{R}^n \to \mathbb{R}^m,
        x \mapsto (\alpha f)(x) = \alpha f(x).

    Parameters
    ----------
    function
        The function :math:`f`.
    scalar
        The scalar :math:`\alpha`.
    """

    def __init__(self, function: Function, scalar: ScalarLike):
        if not isinstance(function, Function):
            raise TypeError(
                "The function to be scaled must be an object of type `Function`."
            )

        self._function = function
        self._scalar = utils.as_numpy_scalar(scalar)

        super().__init__(
            input_shape=self._function.input_shape,
            output_shape=self._function.output_shape,
        )

    @property
    def function(self) -> Function:
        r"""The function :math:`f`."""
        return self._function

    @property
    def scalar(self) -> ScalarType:
        r"""The scalar :math:`\alpha`."""
        return self._scalar

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._scalar * self._function(x)

    @functools.singledispatchmethod
    def __mul__(self, other):
        if np.ndim(other) == 0:
            return ScaledFunction(
                function=self._function,
                scalar=self._scalar * np.asarray(other),
            )

        return super().__mul__(other)

    @functools.singledispatchmethod
    def __rmul__(self, other):
        if np.ndim(other) == 0:
            return ScaledFunction(
                function=self._function,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)
