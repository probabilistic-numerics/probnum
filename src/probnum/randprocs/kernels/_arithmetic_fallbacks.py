"""Fallback-implementations of Kernel arithmetic."""

from __future__ import annotations

import functools
import operator
from typing import Optional, Tuple, Union

import numpy as np

from probnum import utils
from probnum.typing import NotImplementedType, ScalarLike

from ._kernel import BinaryOperandType, Kernel

########################################################################################
# Generic Linear Operator Arithmetic (Fallbacks)
########################################################################################


class ScaledKernel(Kernel):
    r"""Kernel scaled with a (positive) scalar.

    Define a new kernel

    .. math ::
        k(x_0, x_1) = o k'(x_0, x_1)

    by scaling with a positive scalar :math:`o \in (0, \infty)`.

    Parameters
    ----------
    kernel
        Kernel.
    scalar
        Scalar to multiply with.
    """

    def __init__(self, kernel: Kernel, scalar: ScalarLike):

        if not isinstance(kernel, Kernel):
            raise TypeError("`kernel` must be a `Kernel`")

        if np.ndim(scalar) != 0:
            raise TypeError("`scalar` must be a scalar.")

        if not scalar > 0.0:
            raise ValueError("'scalar' must be positive.")

        self._kernel = kernel
        self._scalar = utils.as_numpy_scalar(scalar)

        super().__init__(
            input_shape=kernel.input_shape, output_shape=kernel.output_shape
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:
        return self._scalar * self._kernel(x0, x1)

    def __repr__(self) -> str:
        return f"{self._scalar} * {self._kernel}"


class SumKernel(Kernel):
    r"""Sum of kernels.

    Define a new kernel

    .. math ::
        k(x_0, x_1) = \sum_{i=1}^m k_i(x_0, x_1)

    from a set of kernels :math:`k_i` via summation.

    Parameters
    ----------
    summands
        Kernels to sum together. Must have the same ``input_shape`` and
        ``output_shape``.
    """

    def __init__(self, *summands: Kernel):

        if not all(
            (summand.input_shape == summands[0].input_shape)
            and (summand.output_shape == summands[0].output_shape)
            for summand in summands
        ):
            raise ValueError("All summands must have the same in- and output shape.")

        self._summands = SumKernel._expand_sum_kernels(*summands)

        super().__init__(
            input_shape=summands[0].input_shape, output_shape=summands[0].output_shape
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand(x0, x1) for summand in self._summands)
        )

    def __repr__(self):
        res = "SumKernel [\n"
        for s in self._summands:
            res += f"\t{s}, \n"
        return res + "]"

    @staticmethod
    def _expand_sum_kernels(*summands: Kernel) -> Tuple[Kernel, ...]:
        expanded_summands = []

        for summand in summands:
            if isinstance(summand, SumKernel):
                # pylint: disable="protected-access"
                expanded_summands.extend(summand._summands)
            else:
                expanded_summands.append(summand)

        return tuple(expanded_summands)


class ProductKernel(Kernel):
    r"""(Element-wise) Product of kernels.

    Define a new kernel

    .. math ::
        k(x_0, x_1) = \prod_{i=1}^m k_i(x_0, x_1)

    from a set of kernels :math:`k_i` via multiplication.

    Parameters
    ----------
    factors
        Kernels to multiply together. Must have the same ``input_shape`` and
        ``output_shape``.
    """

    def __init__(self, *factors: Kernel):

        if not all(
            (factor.input_shape == factors[0].input_shape)
            and (factor.output_shape == factors[0].output_shape)
            for factor in factors
        ):
            raise ValueError("All factors must have the same in- and output shape.")

        self._factors = ProductKernel._expand_prod_kernels(*factors)

        super().__init__(
            input_shape=factors[0].input_shape, output_shape=factors[0].output_shape
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return functools.reduce(
            operator.mul, (factor(x0, x1) for factor in self._factors)
        )

    def __repr__(self):
        res = "ProductKernel [\n"
        for s in self._factors:
            res += f"\t{s}, \n"
        return res + "]"

    @staticmethod
    def _expand_prod_kernels(*factors: Kernel) -> Tuple[Kernel, ...]:
        expanded_factors = []

        for factor in factors:
            if isinstance(factor, ProductKernel):
                # pylint: disable="protected-access"
                expanded_factors.extend(factor._factors)
            else:
                expanded_factors.append(factor)

        return tuple(expanded_factors)


def _mul_fallback(
    op1: BinaryOperandType, op2: BinaryOperandType
) -> Union[Kernel, NotImplementedType]:
    res = NotImplemented

    if isinstance(op1, Kernel):
        if isinstance(op2, Kernel):
            res = ProductKernel(op1, op2)
        elif np.ndim(op2) == 0:
            res = ScaledKernel(kernel=op1, scalar=op2)
    elif isinstance(op2, Kernel):
        if np.ndim(op1) == 0:
            res = ScaledKernel(kernel=op2, scalar=op1)
    return res
