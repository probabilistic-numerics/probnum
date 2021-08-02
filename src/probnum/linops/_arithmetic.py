"""Linear operator arithmetic."""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse

from probnum.typing import NotImplementedType, ScalarArgType, ShapeArgType

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
    Embedding,
    Identity,
    LinearOperator,
    Matrix,
    Selection,
    TransposedLinearOperator,
    _ConjugateLinearOperator,
    _InverseLinearOperator,
    _TypeCastLinearOperator,
)
from ._scaling import Scaling, Zero

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
    Selection,
    Embedding,
    Zero,
    Kronecker,
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
    [LinearOperator, LinearOperator], Union[LinearOperator, NotImplementedType]
]
_BinaryOperatorRegistryType = Dict[Tuple[type, type], _BinaryOperatorType]


_add_fns: _BinaryOperatorRegistryType = {}
_sub_fns: _BinaryOperatorRegistryType = {}
_mul_fns: _BinaryOperatorRegistryType = {}
_matmul_fns: _BinaryOperatorRegistryType = {}

########################################################################################
# Fill Arithmetics Registries
########################################################################################

# Scaling
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
_add_fns[(Scaling, Scaling)] = Scaling._add_scaling
_sub_fns[(Scaling, Scaling)] = Scaling._sub_scaling
_mul_fns[(Scaling, Scaling)] = Scaling._mul_scaling
_matmul_fns[(Scaling, Scaling)] = Scaling._matmul_scaling

# ScaledLinearOperator
def _matmul_scaled_op(scaled, anylinop):
    return scaled._scalar * (scaled._linop @ anylinop)


def _matmul_op_scaled(anylinop, scaled):
    return scaled._scalar * (anylinop @ scaled._linop)


for op_type in _AnyLinOp:
    _matmul_fns[(ScaledLinearOperator, op_type)] = _matmul_scaled_op
    _matmul_fns[(op_type, ScaledLinearOperator)] = _matmul_op_scaled
    _matmul_fns[(NegatedLinearOperator, op_type)] = _matmul_scaled_op
    _matmul_fns[(op_type, NegatedLinearOperator)] = _matmul_op_scaled


# Kronecker


def _matmul_scaling_kronecker(scaling: Scaling, kronecker: Kronecker) -> Kronecker:
    if scaling.is_isotropic:
        return scaling.scalar * kronecker
    return NotImplemented


def _matmul_kronecker_scaling(kronecker: Kronecker, scaling: Scaling) -> Kronecker:
    if scaling.is_isotropic:
        return kronecker * scaling.scalar
    return NotImplemented


def _mul_scalar_kronecker(scalar: ScalarArgType, kronecker: Kronecker) -> Kronecker:

    prefer_A = ("scalar", type(kronecker.A)) in _mul_fns
    prefer_B = ("scalar", type(kronecker.B)) in _mul_fns

    if prefer_A and not prefer_B:
        return Kronecker(A=scalar * kronecker.A, B=kronecker.B)

    if prefer_B and not prefer_A:
        return Kronecker(A=kronecker.A, B=scalar * kronecker.B)

    return Kronecker(A=scalar * kronecker.A, B=kronecker.B)


def _mul_kronecker_scalar(kronecker: Kronecker, scalar: ScalarArgType) -> Kronecker:

    prefer_A = (type(kronecker.A), "scalar") in _mul_fns
    prefer_B = (type(kronecker.B), "scalar") in _mul_fns

    if prefer_A and not prefer_B:
        return Kronecker(A=kronecker.A * scalar, B=kronecker.B)

    if prefer_B and not prefer_A:
        return Kronecker(A=kronecker.A, B=kronecker.B * scalar)

    return Kronecker(A=kronecker.A, B=kronecker.B * scalar)


_matmul_fns[(Kronecker, Kronecker)] = Kronecker._matmul_kronecker
_mul_fns[(Kronecker, Kronecker)] = Kronecker._mul_kronecker
_add_fns[(Kronecker, Kronecker)] = Kronecker._add_kronecker
_sub_fns[(Kronecker, Kronecker)] = Kronecker._sub_kronecker

_mul_fns[("scalar", Kronecker)] = _mul_scalar_kronecker
_mul_fns[(Kronecker, "scalar")] = _mul_kronecker_scalar
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


def _mul_matrix_scalar(mat, scalar) -> Union[NotImplementedType, Matrix]:
    if np.isscalar(scalar):
        return Matrix(A=scalar * mat.A)

    return NotImplemented


def _mul_scalar_matrix(scalar, mat) -> Union[NotImplementedType, Matrix]:
    if np.isscalar(scalar):
        return Matrix(A=scalar * mat.A)

    return NotImplemented


_mul_fns[(Matrix, "scalar")] = _mul_matrix_scalar
_mul_fns[("scalar", Matrix)] = _mul_scalar_matrix
_matmul_fns[(Matrix, Matrix)] = Matrix._matmul_matrix
_matmul_fns[(Scaling, Matrix)] = _matmul_scaling_matrix
_matmul_fns[(Matrix, Scaling)] = _matmul_matrix_scaling


_matmul_fns[(Selection, Matrix)] = lambda sel, mat: Matrix(sel @ mat.A)
_matmul_fns[(Embedding, Matrix)] = lambda emb, mat: Matrix(emb @ mat.A)
_matmul_fns[(Matrix, Selection)] = lambda mat, sel: Matrix(mat.A @ sel)
_matmul_fns[(Matrix, Embedding)] = lambda mat, emb: Matrix(mat.A @ emb)

_add_fns[(Matrix, Matrix)] = lambda mat1, mat2: Matrix(mat1.A + mat2.A)
_sub_fns[(Matrix, Matrix)] = lambda mat1, mat2: Matrix(mat1.A - mat2.A)

_matmul_fns[(Matrix, _InverseLinearOperator)] = lambda mat, inv: Matrix(mat.A @ inv)
_matmul_fns[(_InverseLinearOperator, Matrix)] = lambda inv, mat: Matrix(inv @ mat.A)


# Identity
for op_type in _AnyLinOp:
    _matmul_fns[(Identity, op_type)] = lambda idty, other: other
    _matmul_fns[(op_type, Identity)] = lambda other, idty: other


# Selection / Embedding
def _matmul_selection_embedding(
    selection: Selection, embedding: Embedding
) -> Union[NotImplementedType, Identity]:

    if (embedding.shape[-1] == selection.shape[-2]) and np.all(
        selection.indices == embedding._put_indices
    ):
        return Identity(shape=(selection.shape[-2], embedding.shape[-1]))

    return NotImplemented


_matmul_fns[(Selection, Embedding)] = _matmul_selection_embedding
# Embedding @ Selection would be Projection

# Zero
def _matmul_zero_anylinop(z: Zero, op: LinearOperator) -> Zero:
    if z.shape[1] != op.shape[0]:
        raise ValueError(f"shape mismatch")  # TODO

    return Zero(shape=(z.shape[0], op.shape[1]))


def _matmul_anylinop_zero(op: LinearOperator, z: Zero) -> Zero:
    if z.shape[0] != op.shape[1]:
        raise ValueError(f"shape mismatch")  # TODO

    return Zero(shape=(op.shape[0], z.shape[1]))


def _add_zero_anylinop(z: Zero, op: LinearOperator) -> Zero:
    if z.shape != op.shape:
        raise ValueError(f"shape mismatch")  # TODO

    return op


def _add_anylinop_zero(op: LinearOperator, z: Zero) -> Zero:
    if z.shape != op.shape:
        raise ValueError(f"shape mismatch")  # TODO

    return op


def _sub_zero_anylinop(z: Zero, op: LinearOperator) -> Zero:
    if z.shape != op.shape:
        raise ValueError(f"shape mismatch")  # TODO

    return -op


def _sub_anylinop_zero(op: LinearOperator, z: Zero) -> Zero:
    if z.shape != op.shape:
        raise ValueError(f"shape mismatch")  # TODO

    return op


for op_type in _AnyLinOp:
    _matmul_fns[(Zero, op_type)] = _matmul_zero_anylinop
    _matmul_fns[(op_type, Zero)] = _matmul_anylinop_zero
    _add_fns[(Zero, op_type)] = _add_zero_anylinop
    _add_fns[(op_type, Zero)] = _add_anylinop_zero
    _sub_fns[(Zero, op_type)] = _sub_zero_anylinop
    _sub_fns[(op_type, Zero)] = _sub_anylinop_zero


########################################################################################
# Apply
########################################################################################


def _apply(
    op_registry: _BinaryOperatorRegistryType,
    op1: LinearOperator,
    op2: LinearOperator,
    fallback_operator: Optional[
        Callable[
            [LinearOperator, LinearOperator],
            Union[LinearOperator, NotImplementedType],
        ]
    ] = None,
) -> Union[LinearOperator, NotImplementedType]:
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
