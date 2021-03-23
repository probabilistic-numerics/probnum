import functools
import operator
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse

import probnum.utils
from probnum.type import ScalarArgType, ShapeArgType

from ._linear_operator import BinaryOperandType, LinearOperator, MatrixMult, ScalarMult


def add(op1: BinaryOperandType, op2: BinaryOperandType) -> LinearOperator:
    op1, op2 = _operands_to_compatible_linops(op1, op2)

    return _apply(_add_fns, op1, op2, fallback_operator=SumLinearOperator)


def sub(op1: BinaryOperandType, op2: BinaryOperandType) -> LinearOperator:
    op1, op2 = _operands_to_compatible_linops(op1, op2)

    return _apply(
        _sub_fns,
        op1,
        op2,
        fallback_operator=lambda op1, op2: SumLinearOperator(op1, -op2),
    )


def mul(op1: BinaryOperandType, op2: BinaryOperandType) -> LinearOperator:
    op1, op2 = _operands_to_compatible_linops(op1, op2)

    return _apply(_mul_fns, op1, op2, fallback_operator=_mul_fallback)


def matmul(op1: LinearOperator, op2: LinearOperator) -> LinearOperator:
    return _apply(_matmul_fns, op1, op2, fallback_operator=ProductLinearOperator)


########################################################################################
# Operator registry
########################################################################################

_BinaryOperatorType = Callable[
    [LinearOperator, LinearOperator], Union[LinearOperator, type(NotImplemented)]
]
_BinaryOperatorRegistryType = Dict[Tuple[type, type], _BinaryOperatorType]


_add_fns: _BinaryOperatorRegistryType = {}
_sub_fns: _BinaryOperatorRegistryType = {}
_mul_fns: _BinaryOperatorRegistryType = {}
_matmul_fns: _BinaryOperatorRegistryType = {}


def _apply(
    op_registry: _BinaryOperatorRegistryType,
    op1: LinearOperator,
    op2: LinearOperator,
    fallback_operator: Optional[
        Callable[
            [LinearOperator, LinearOperator],
            Union[LinearOperator, type(NotImplemented)],
        ]
    ] = None,
) -> Union[LinearOperator, type(NotImplemented)]:
    key = (type(op1), type(op2))

    if key in op_registry:
        res = op_registry[key](op1, op2)
    else:
        res = NotImplemented

    if res is NotImplemented and fallback_operator is not None:
        res = fallback_operator(op1, op2)

    return res


########################################################################################
# Helper Functions
########################################################################################


def _operand_to_linop(operand: Any, shape: ShapeArgType) -> Optional[LinearOperator]:
    if isinstance(operand, LinearOperator):
        pass
    elif np.ndim(operand) == 0:
        operand = ScalarMult(shape, operand)
    elif isinstance(operand, (np.ndarray, scipy.sparse.spmatrix)):
        operand = MatrixMult(operand)
    else:
        operand = None

    return operand


def _operands_to_compatible_linops(
    op1: Any, op2: Any
) -> Tuple[Optional[LinearOperator], Optional[LinearOperator]]:
    if not isinstance(op1, LinearOperator) and not isinstance(op2, LinearOperator):
        raise TypeError("At least one of the two operands must be a `LinearOperator`.")
    elif not isinstance(op1, LinearOperator):
        op1 = _operand_to_linop(op1, shape=op2.shape)
    elif not isinstance(op2, LinearOperator):
        op2 = _operand_to_linop(op2, shape=op1.shape)
    else:
        # TODO: check whether both have the same shape
        pass

    return op1, op2


########################################################################################
# Generic Linear Operator Arithmetic (Fallbacks)
########################################################################################


class ScaledLinearOperator(LinearOperator):
    """Linear operator scaled with a scalar."""

    def __init__(self, linop: LinearOperator, scalar: ScalarArgType):
        if not isinstance(linop, LinearOperator):
            raise TypeError("`linop` must be a `LinearOperator`")

        if not np.isscalar(scalar):
            raise TypeError("`scalar` must be a scalar.")

        dtype = np.result_type(linop.dtype, scalar)

        self._linop = linop
        self._scalar = probnum.utils.as_numpy_scalar(scalar, dtype)

        super().__init__(dtype, self._linop.shape)

    def _matvec(self, vec: np.ndarray) -> np.ndarray:
        return self._scalar * self._linop.matvec(vec)

    def _rmatvec(self, vec: np.ndarray) -> np.ndarray:
        return np.conj(self._scalar) * self._linop.rmatvec(vec)

    def _matmat(self, mat: np.ndarray) -> np.ndarray:
        return self._scalar * self._linop.matmat(mat)

    def _rmatmat(self, mat: np.ndarray) -> np.ndarray:
        return np.conj(self._scalar) * self._linop.rmatmat(mat)

    def _adjoint(self) -> LinearOperator:
        return self._linop.H * np.conj(self._scalar)

    def todense(self) -> np.ndarray:
        return self._scalar * self._linop.todense()

    def inv(self) -> "ScaledLinearOperator":
        return ScaledLinearOperator(self._linop.inv(), 1.0 / self._scalar)

    def trace(self):
        return self._scalar * self._linop.trace()


class NegatedLinearOperator(ScaledLinearOperator):
    def __init__(self, linop: LinearOperator):
        super().__init__(linop, scalar=probnum.utils.as_numpy_scalar(-1, linop.dtype))

    def __neg__(self) -> "LinearOperator":
        return self._linop


class SumLinearOperator(LinearOperator):
    """Sum of two linear operators."""

    def __init__(self, *summands: LinearOperator):
        if not all(isinstance(summand, LinearOperator) for summand in summands):
            raise TypeError("All summands must be `LinearOperator`s")

        if len(summands) < 2:
            raise ValueError("There must be at least two summands")

        if not all(summand.shape == summands[0].shape for summand in summands):
            raise ValueError("All summands must have the same shape")

        self._summands = SumLinearOperator._expand_sum_ops(*summands)

        super().__init__(
            dtype=np.find_common_type(
                [summand.dtype for summand in self._summands], []
            ),
            shape=summands[0].shape,
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand.matvec(x) for summand in self._summands)
        )

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand.rmatvec(x) for summand in self._summands)
        )

    def _matmat(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand.matmat(x) for summand in self._summands)
        )

    def _rmatmat(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand.rmatmat(x) for summand in self._summands)
        )

    def __neg__(self):
        return SumLinearOperator(*(-summand for summand in self._summands))

    def todense(self) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand.todense() for summand in self._summands)
        )

    def trace(self) -> np.floating:
        return functools.reduce(
            operator.add, (summand.trace() for summand in self._summands)
        )

    @staticmethod
    def _expand_sum_ops(*summands: LinearOperator) -> Tuple[LinearOperator, ...]:
        expanded_summands = []

        for summand in summands:
            if isinstance(summand, SumLinearOperator):
                expanded_summands.extend(summand._summands)
            else:
                expanded_summands.append(summand)

        return tuple(expanded_summands)


def _mul_fallback(
    op1: LinearOperator, op2: LinearOperator
) -> Union[LinearOperator, type(NotImplemented)]:
    res = NotImplemented

    if isinstance(op1, ScalarMult):
        res = ScaledLinearOperator(op2, op1._scalar)
    elif isinstance(op2, ScalarMult):
        res = ScaledLinearOperator(op1, op2._scalar)

    return res


class ProductLinearOperator(LinearOperator):
    """(Operator) Product of two linear operators."""

    def __init__(self, *factors: LinearOperator):
        if not all(isinstance(factor, LinearOperator) for factor in factors):
            raise TypeError("All factors must be `LinearOperator`s")

        if len(factors) < 2:
            raise ValueError("There must be at least two factors")

        if not all(
            lfactor.shape[1] == rfactor.shape[0]
            for lfactor, rfactor in zip(factors[:-1], factors[1:])
        ):
            raise ValueError(
                f"Shape mismatch: Cannot multiply linear operators with shapes "
                f"{' x '.join(factor.shape for factor in factors)}."
            )

        self._factors = ProductLinearOperator._expand_prod_ops(*factors)

        super().__init__(
            dtype=np.find_common_type([factor.dtype for factor in self._factors], []),
            shape=(self._factors[0].shape[0], self._factors[-1].shape[1]),
        )

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            lambda vec, op: op.matvec(vec), reversed(self._factors), x
        )

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(lambda vec, op: op.rmatvec(vec), self._factors, x)

    def _matmat(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            lambda vec, op: op.matmat(vec), reversed(self._factors), x
        )

    def _rmatmat(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(lambda vec, op: op.rmatmat(vec), self._factors, x)

    def todense(self) -> np.ndarray:
        return functools.reduce(
            operator.matmul, (factor.todense() for factor in self._factors)
        )

    def inv(self) -> np.ndarray:
        return ProductLinearOperator(
            *(factor.inv() for factor in reversed(self._factors))
        )

    def trace(self) -> np.floating:
        return np.trace(self.todense())

    @staticmethod
    def _expand_prod_ops(*factors: LinearOperator) -> Tuple[LinearOperator, ...]:
        expanded_factors = []

        for factor in factors:
            if isinstance(factor, ProductLinearOperator):
                expanded_factors.extend(factor._factors)
            else:
                expanded_factors.append(factor)

        return tuple(expanded_factors)


class TransposedLinearOperator(LinearOperator):
    """Transposition of a linear operator."""

    def __init__(self, linop: LinearOperator):
        self._linop = linop

        super().__init__(shape=(linop.shape[1], linop.shape[0]), dtype=linop.dtype)

    def _matvec(self, x):
        # NB. np.conj works also on sparse matrices
        return np.conj(self._linop._rmatvec(np.conj(x)))

    def _rmatvec(self, x):
        return np.conj(self._linop._matvec(np.conj(x)))

    def _matmat(self, x):
        # NB. np.conj works also on sparse matrices
        return np.conj(self._linop._rmatmat(np.conj(x)))

    def _rmatmat(self, x):
        return np.conj(self._linop._matmat(np.conj(x)))

    def todense(self):
        return self._linop.todense().T

    def inv(self):
        return self._linop.inv().T


class InverseLinearOperator(MatrixMult):
    def __init__(self, linop: LinearOperator):
        self._linop = linop

        super().__init__(A=np.inv(self._linop.todense()))

    def inv(self) -> LinearOperator:
        return self._linop
