"""Fallback-implementations of covariance function arithmetic."""

from __future__ import annotations

import functools
import operator
from typing import Optional, Tuple, Union

import numpy as np

from probnum import linops, utils
from probnum.typing import NotImplementedType, ScalarLike

from ._covariance_function import BinaryOperandType, CovarianceFunction

########################################################################################
# Generic Linear Operator Arithmetic (Fallbacks)
########################################################################################


class ScaledCovarianceFunction(CovarianceFunction):
    r"""Covariance function scaled with a (positive) scalar.

    Define a new covariance function

    .. math ::
        k(x_0, x_1) = o k'(x_0, x_1)

    by scaling with a positive scalar :math:`o \in (0, \infty)`.

    Parameters
    ----------
    covfunc
        Covariance function.
    scalar
        Scalar to multiply with.
    """

    def __init__(self, covfunc: CovarianceFunction, scalar: ScalarLike):
        if not isinstance(covfunc, CovarianceFunction):
            raise TypeError("`covfunc` must be a `CovarianceFunction`")

        if np.ndim(scalar) != 0:
            raise TypeError("`scalar` must be a scalar.")

        self._covfunc = covfunc
        self._scalar = utils.as_numpy_scalar(scalar)

        super().__init__(
            input_shape=covfunc.input_shape,
            output_shape_0=covfunc.output_shape_0,
            output_shape_1=covfunc.output_shape_1,
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray] = None) -> np.ndarray:
        return self._scalar * self._covfunc(x0, x1)

    def _evaluate_linop(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> linops.LinearOperator:
        return self._scalar * self._covfunc.linop(x0, x1)

    def __repr__(self) -> str:
        return f"{self._scalar} * {self._covfunc}"


class SumCovarianceFunction(CovarianceFunction):
    r"""Sum of covariance functions.

    Define a new covariance function

    .. math ::
        k(x_0, x_1) = \sum_{i=1}^m k_i(x_0, x_1)

    from a set of covariance functions :math:`k_i` via summation.

    Parameters
    ----------
    summands
        Covariance functions to sum together. Must have the same ``input_shape`` and
        ``output_shape``.
    """

    def __init__(self, *summands: CovarianceFunction):

        if not all(
            (summand.input_shape == summands[0].input_shape)
            and (summand.output_shape_0 == summands[0].output_shape_0)
            and (summand.output_shape_1 == summands[0].output_shape_1)
            for summand in summands
        ):
            raise ValueError("All summands must have the same in- and output shape.")

        self._summands = SumCovarianceFunction._expand_sum_covfuncs(*summands)

        super().__init__(
            input_shape=summands[0].input_shape,
            output_shape_0=summands[0].output_shape_0,
            output_shape_1=summands[0].output_shape_1,
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand(x0, x1) for summand in self._summands)
        )

    def _evaluate_linop(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> linops.LinearOperator:
        return functools.reduce(
            operator.add, (summand.linop(x0, x1) for summand in self._summands)
        )

    def __repr__(self):
        res = "SumCovarianceFunction [\n\t"
        res += ",\n\t".join(repr(summand) for summand in self._summands)
        res += "\n]"
        return res

    @staticmethod
    def _expand_sum_covfuncs(
        *summands: CovarianceFunction,
    ) -> Tuple[CovarianceFunction, ...]:
        expanded_summands = []

        for summand in summands:
            if isinstance(summand, SumCovarianceFunction):
                # pylint: disable="protected-access"
                expanded_summands.extend(summand._summands)
            else:
                expanded_summands.append(summand)

        return tuple(expanded_summands)


class ProductCovarianceFunction(CovarianceFunction):
    r"""(Element-wise) Product of covariance functions.

    Define a new covariance function

    .. math ::
        k(x_0, x_1) = \prod_{i=1}^m k_i(x_0, x_1)

    from a set of covariance functions :math:`k_i` via multiplication.

    Parameters
    ----------
    factors
        Covariance functions to multiply together. Must have the same ``input_shape``
        and ``output_shape``.
    """

    def __init__(self, *factors: CovarianceFunction):

        if not all(
            (factor.input_shape == factors[0].input_shape)
            and (factor.output_shape_0 == factors[0].output_shape_0)
            and (factor.output_shape_1 == factors[0].output_shape_1)
            for factor in factors
        ):
            raise ValueError("All factors must have the same in- and output shape.")

        self._factors = ProductCovarianceFunction._expand_prod_covfuncs(*factors)

        super().__init__(
            input_shape=factors[0].input_shape,
            output_shape_0=factors[0].output_shape_0,
            output_shape_1=factors[0].output_shape_1,
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return functools.reduce(
            operator.mul, (factor(x0, x1) for factor in self._factors)
        )

    def __repr__(self):
        res = "ProductCovarianceFunction [\n\t"
        res += ",\n\t".join(repr(factor) for factor in self._factors)
        res += "\n]"
        return res

    @staticmethod
    def _expand_prod_covfuncs(
        *factors: CovarianceFunction,
    ) -> Tuple[CovarianceFunction, ...]:
        expanded_factors = []

        for factor in factors:
            if isinstance(factor, ProductCovarianceFunction):
                # pylint: disable="protected-access"
                expanded_factors.extend(factor._factors)
            else:
                expanded_factors.append(factor)

        return tuple(expanded_factors)


def _mul_fallback(
    op1: BinaryOperandType, op2: BinaryOperandType
) -> Union[CovarianceFunction, NotImplementedType]:
    res = NotImplemented

    if isinstance(op1, CovarianceFunction):
        if isinstance(op2, CovarianceFunction):
            res = ProductCovarianceFunction(op1, op2)
        elif np.ndim(op2) == 0:
            res = ScaledCovarianceFunction(op1, scalar=op2)
    elif isinstance(op2, CovarianceFunction):
        if np.ndim(op1) == 0:
            res = ScaledCovarianceFunction(op2, scalar=op1)
    return res
