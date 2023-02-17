"""Fallback-implementations of LinearOperator arithmetic."""
from __future__ import annotations

import functools
import operator
from typing import Tuple, Union

import numpy as np

from probnum.typing import NotImplementedType, ScalarLike
import probnum.utils

from ._linear_operator import BinaryOperandType, LambdaLinearOperator, LinearOperator

########################################################################################
# Generic Linear Operator Arithmetic (Fallbacks)
########################################################################################


class ScaledLinearOperator(LambdaLinearOperator):
    """Linear operator scaled with a scalar."""

    def __init__(self, linop: LinearOperator, scalar: ScalarLike):
        if not isinstance(linop, LinearOperator):
            raise TypeError("`linop` must be a `LinearOperator`")

        if np.ndim(scalar) != 0:
            raise TypeError("`scalar` must be a scalar.")

        dtype = np.result_type(linop.dtype, scalar)

        self._linop = linop
        self._scalar = probnum.utils.as_numpy_scalar(scalar, dtype)

        super().__init__(
            self._linop.shape,
            dtype=dtype,
            matmul=lambda x: self._scalar * (self._linop @ x),
            todense=lambda: self._scalar * self._linop.todense(cache=False),
            transpose=lambda: self._scalar * self._linop.T,
            inverse=self._inv,
            trace=lambda: self._scalar * self._linop.trace(),
        )

        # Matrix properties
        self.is_symmetric = self._linop.is_symmetric
        self.is_lower_triangular = self._linop.is_lower_triangular
        self.is_upper_triangular = self._linop.is_upper_triangular

        if self._linop.is_positive_definite:
            if self._scalar > 0:
                self.is_positive_definite = True
            else:
                self.is_positive_definite = False

    def _solve(self, B: np.ndarray) -> np.ndarray:
        return self.inv() @ B

    def _inv(self) -> "ScaledLinearOperator":
        if self._scalar == 0:
            raise np.linalg.LinAlgError("The operator is not invertible")

        return ScaledLinearOperator(self._linop.inv(), 1.0 / self._scalar)

    def __repr__(self) -> str:
        return f"{self._scalar} * {self._linop}"

    def _symmetrize(self) -> ScaledLinearOperator:
        return ScaledLinearOperator(
            linop=self._linop.symmetrize(),
            scalar=self._scalar,
        )


class NegatedLinearOperator(ScaledLinearOperator):
    """Negation of a given linear operator."""

    def __init__(self, linop: LinearOperator):
        super().__init__(linop, scalar=probnum.utils.as_numpy_scalar(-1, linop.dtype))

    def __neg__(self) -> "LinearOperator":
        return self._linop

    def __repr__(self) -> str:
        return f"-{self._linop}"


class SumLinearOperator(LambdaLinearOperator):
    """Sum of linear operators."""

    def __init__(self, *summands: LinearOperator):

        if not all(summand.shape == summands[0].shape for summand in summands):
            raise ValueError("All summands must have the same shape.")

        self._summands = SumLinearOperator._expand_sum_ops(*summands)

        super().__init__(
            shape=summands[0].shape,
            dtype=np.find_common_type(
                [summand.dtype for summand in self._summands], []
            ),
            matmul=lambda x: functools.reduce(
                operator.add, (summand @ x for summand in self._summands)
            ),
            todense=lambda: functools.reduce(
                operator.add,
                (summand.todense(cache=False) for summand in self._summands),
            ),
            transpose=lambda: SumLinearOperator(
                *(summand.T for summand in self._summands)
            ),
            trace=lambda: functools.reduce(
                operator.add, (summand.trace() for summand in self._summands)
            ),
        )

        # Matrix properties
        if all(summand.is_symmetric for summand in self._summands):
            self.is_symmetric = True

        if all(summand.is_lower_triangular for summand in self._summands):
            self.is_lower_triangular = True

        if all(summand.is_upper_triangular for summand in self._summands):
            self.is_upper_triangular = True

        if all(summand.is_positive_definite for summand in self._summands):
            self.is_positive_definite = True

    @property
    def summands(self) -> Tuple[LinearOperator, ...]:
        """The summands that make up this sum of :class:`LinearOperator`s."""
        return self._summands

    def __neg__(self):
        return SumLinearOperator(*(-summand for summand in self._summands))

    def __repr__(self):
        res = "SumLinearOperator [\n\t"
        res += ",\n\t".join(repr(summand) for summand in self._summands)
        res += "\n]"
        return res

    @staticmethod
    def _expand_sum_ops(*summands: LinearOperator) -> Tuple[LinearOperator, ...]:
        expanded_summands = []

        for summand in summands:
            if isinstance(summand, SumLinearOperator):
                expanded_summands.extend(summand.summands)
            else:
                expanded_summands.append(summand)

        return tuple(expanded_summands)

    def _symmetrize(self) -> SumLinearOperator:
        return SumLinearOperator(*[summand.symmetrize() for summand in self._summands])


def _mul_fallback(
    op1: BinaryOperandType, op2: BinaryOperandType
) -> Union[LinearOperator, NotImplementedType]:
    res = NotImplemented

    if isinstance(op1, LinearOperator):
        if np.ndim(op2) == 0:
            res = ScaledLinearOperator(op1, op2)
    elif isinstance(op2, LinearOperator):
        if np.ndim(op1) == 0:
            res = ScaledLinearOperator(op2, op1)
    return res


class ProductLinearOperator(LambdaLinearOperator):
    """(Operator) Product of linear operators."""

    def __init__(self, *factors: LinearOperator):

        if not all(
            lfactor.shape[1] == rfactor.shape[0]
            for lfactor, rfactor in zip(factors[:-1], factors[1:])
        ):
            raise ValueError(
                f"Shape mismatch: Cannot multiply linear operators with shapes "
                f"{' x '.join(str(factor.shape) for factor in factors)}."
            )

        self._factors = ProductLinearOperator._expand_prod_ops(*factors)

        super().__init__(
            shape=(self._factors[0].shape[0], self._factors[-1].shape[1]),
            dtype=np.find_common_type([factor.dtype for factor in self._factors], []),
            matmul=lambda x: functools.reduce(
                lambda vec, op: op @ vec, reversed(self._factors), x
            ),
            todense=lambda: functools.reduce(
                operator.matmul,
                (factor.todense(cache=False) for factor in self._factors),
            ),
            transpose=lambda: ProductLinearOperator(
                *(factor.T for factor in reversed(self._factors))
            ),
            inverse=lambda: ProductLinearOperator(
                *(factor.inv() for factor in reversed(self._factors))
            ),
            det=lambda: functools.reduce(
                operator.mul, (factor.det() for factor in self._factors)
            ),
            logabsdet=lambda: functools.reduce(
                operator.add, (factor.logabsdet() for factor in self._factors)
            ),
        )

    @property
    def factors(self) -> Tuple[LinearOperator, ...]:
        """The factors that make up this product of :class:`LinearOperator`s."""
        return self._factors

    @staticmethod
    def _expand_prod_ops(*factors: LinearOperator) -> Tuple[LinearOperator, ...]:
        expanded_factors = []

        for factor in factors:
            if isinstance(factor, ProductLinearOperator):
                expanded_factors.extend(factor.factors)
            else:
                expanded_factors.append(factor)

        return tuple(expanded_factors)

    def __repr__(self):
        res = "ProductLinearOperator [\n\t"
        res += ",\n\t".join(repr(factor) for factor in self._factors)
        res += "\n]"
        return res

    def _solve(self, B: np.ndarray) -> np.ndarray:
        return functools.reduce(lambda b, op: op.solve(b), self._factors, B)


def _matmul_fallback(
    op1: BinaryOperandType, op2: BinaryOperandType
) -> Union[LinearOperator, NotImplementedType]:
    res = NotImplemented

    if isinstance(op1, LinearOperator) and isinstance(op2, LinearOperator):
        res = ProductLinearOperator(op1, op2)

    return res
