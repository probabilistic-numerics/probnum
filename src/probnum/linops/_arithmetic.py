"""Linear operator arithmetic."""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse

from probnum.typing import ScalarArgType, ShapeArgType

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

for op_type in _AnyLinOp:
    _add_fns[(op_type, Scaling)] = (
        lambda op, scaling: op if scaling._scalar == 0.0 else NotImplemented
    )
    _add_fns[(Scaling, op_type)] = (
        lambda scaling, op: op if scaling._scalar == 0.0 else NotImplemented
    )
    _sub_fns[(Scaling, op_type)] = (
        lambda scaling, op: op if scaling._scalar == 0.0 else NotImplemented
    )
    _sub_fns[(op_type, Scaling)] = (
        lambda op, scaling: op if scaling._scalar == 0.0 else NotImplemented
    )


def _mul_scalar_scaling(scalar: ScalarArgType, scaling: Scaling) -> Scaling:
    if scaling.is_isotropic:
        return Scaling(scalar * scaling.scalar, shape=scaling.shape)

    return Scaling(scalar * scaling.factors, shape=scaling.shape)


def _mul_scaling_scalar(scaling: Scaling, scalar: ScalarArgType) -> Scaling:
    if scaling.is_isotropic:
        return Scaling(scalar * scaling.scalar, shape=scaling.shape)

    return Scaling(scalar * scaling.factors, shape=scaling.shape)


_mul_fns[("scalar", Scaling)] = _mul_scalar_scaling
_mul_fns[(Scaling, "scalar")] = _mul_scaling_scalar

# ScaledLinearOperator
def _matmul_scaled_op(scaled, anylinop):
    return scaled._scalar * (scaled._linop @ anylinop)


def _matmul_op_scaled(anylinop, scaled):
    return scaled._scalar * (anylinop @ scaled._linop)


for op_type in _AnyLinOp:
    _matmul_fns[(ScaledLinearOperator, op_type)] = _matmul_scaled_op
    _matmul_fns[(op_type, ScaledLinearOperator)] = _matmul_op_scaled


# Kronecker

_add_fns[(Kronecker, Kronecker)] = Kronecker._add_kronecker
_sub_fns[(Kronecker, Kronecker)] = Kronecker._sub_kronecker


def _matmul_scaling_kronecker(scaling: Scaling, kronecker: Kronecker) -> Kronecker:
    if scaling.is_isotropic:
        return Kronecker(A=scaling.scalar * kronecker.A, B=kronecker.B)
    return NotImplemented


def _matmul_kronecker_scaling(kronecker: Kronecker, scaling: Scaling) -> Kronecker:
    if scaling.is_isotropic:
        return Kronecker(A=kronecker.A, B=kronecker.B * scaling.scalar)
    return NotImplemented


_matmul_fns[(Kronecker, Kronecker)] = Kronecker._matmul_kronecker
_mul_fns[(Kronecker, Kronecker)] = Kronecker._mul_kronecker

_mul_fns[("scalar", Kronecker)] = lambda sc, kr: Kronecker(A=sc * kr.A, B=kr.B)
_mul_fns[(Kronecker, "scalar")] = lambda kr, sc: Kronecker(A=kr.A, B=kr.B * sc)
_matmul_fns[(Kronecker, Scaling)] = _matmul_kronecker_scaling
_matmul_fns[(Scaling, Kronecker)] = _matmul_scaling_kronecker

# Matrix
def _matmul_scaling_matrix(scaling: Scaling, matrix: Matrix) -> Matrix:
    if scaling.shape[1] != matrix.shape[0]:
        return NotImplemented

    return Matrix(A=np.multiply(scaling.factors[:, np.newaxis], matrix.A))


def _matmul_matrix_scaling(matrix: Matrix, scaling: Scaling) -> Matrix:
    if matrix.shape[1] != scaling.shape[0]:
        return NotImplemented

    return Matrix(A=np.multiply(matrix.A, scaling.factors))


def _mul_matrix_scalar(mat, scalar) -> Union[type(NotImplemented), Matrix]:
    if np.isscalar(scalar):
        return Matrix(A=scalar * mat.A)

    return NotImplemented


def _mul_scalar_matrix(scalar, mat) -> Union[type(NotImplemented), Matrix]:
    if np.isscalar(scalar):
        return Matrix(A=scalar * mat.A)

    return NotImplemented


_mul_fns[(Matrix, "scalar")] = _mul_matrix_scalar
_mul_fns[("scalar", Matrix)] = _mul_scalar_matrix

_matmul_fns[(Matrix, Matrix)] = Matrix._matmul_matrix
_matmul_fns[(Scaling, Matrix)] = _matmul_scaling_matrix
_matmul_fns[(Matrix, Scaling)] = _matmul_matrix_scaling


# Identity
for op_type in _AnyLinOp:
    _matmul_fns[(Identity, op_type)] = lambda idty, other: other
    _matmul_fns[(op_type, Identity)] = lambda other, idty: other

_mul_fns[(Identity, "scalar")] = lambda idty, sc: Scaling(sc, shape=idty.shape)
_mul_fns[("scalar", Identity)] = lambda sc, idty: Scaling(sc, shape=idty.shape)


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
    if np.isscalar(op1):
        key1 = "scalar"
    else:
        key1 = type(op1)

    if np.isscalar(op2):
        key2 = "scalar"
    else:
        key2 = type(op2)

    key = (key1, key2)

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
