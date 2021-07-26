"""Linear operator arithmetic."""
import numbers
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse

from probnum.typing import ShapeArgType

from ._arithmetic_fallbacks import (
    NegatedLinearOperator,
    ProductLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    _matmul_fallback,
    _mul_fallback,
)
from ._kronecker import Kronecker, SymmetricKronecker, Symmetrize
from ._linear_operator import (
    AdjointLinearOperator,
    BinaryOperandType,
    Identity,
    LinearOperator,
    Matrix,
    TransposedLinearOperator,
    _ConjugateLinearOperator,
    _InverseLinearOperator,
    _TypeCastLinearOperator,
)
from ._scaling import Scaling

_AnyLinOp = [
    NegatedLinearOperator,
    ProductLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    AdjointLinearOperator,
    Identity,
    LinearOperator,
    Matrix,
    TransposedLinearOperator,
    SymmetricKronecker,
    Symmetrize,
    _ConjugateLinearOperator,
    _InverseLinearOperator,
    _TypeCastLinearOperator,
    Scaling,
]


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
    return _apply(_mul_fns, op1, op2, fallback_operator=_mul_fallback)


def matmul(op1: LinearOperator, op2: LinearOperator) -> LinearOperator:
    return _apply(_matmul_fns, op1, op2, fallback_operator=_matmul_fallback)


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

# Scaling
_add_fns[(Scaling, Scaling)] = Scaling._add_scaling
_sub_fns[(Scaling, Scaling)] = Scaling._sub_scaling
_mul_fns[(Scaling, Scaling)] = Scaling._mul_scaling
_matmul_fns[(Scaling, Scaling)] = Scaling._matmul_scaling

# Kronecker
_matmul_fns[(Kronecker, Kronecker)] = Kronecker._matmul_kronecker
_mul_fns[(Kronecker, Kronecker)] = Kronecker._mul_kronecker

# Identity
for op_type in _AnyLinOp:
    _matmul_fns[(Identity, op_type)] = lambda idty, other: other
    _matmul_fns[(op_type, Identity)] = lambda other, idty: other


def _mul_id(arg1, arg2):
    if isinstance(arg1, Identity):
        if (
            isinstance(arg2, (int, float, complex, np.number, numbers.Number))
            and np.ndim(arg2) == 0
        ):
            return Scaling(factors=arg2, shape=arg1.shape, dtype=arg1.dtype)

    if isinstance(arg2, Identity):
        if (
            isinstance(arg1, (int, float, complex, np.number, numbers.Number))
            and np.ndim(arg1) == 0
        ):
            return Scaling(factors=arg1, shape=arg2.shape, dtype=arg2.dtype)

    return NotImplemented


for sc_type in [int, float, complex, np.number, numbers.Number]:
    _mul_fns[(Identity, sc_type)] = _mul_id
    _mul_fns[(sc_type, Identity)] = _mul_id


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
        operand = Scaling(operand, shape=shape)
    elif isinstance(operand, (np.ndarray, scipy.sparse.spmatrix)):
        operand = Matrix(operand)
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
