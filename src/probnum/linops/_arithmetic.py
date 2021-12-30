"""Linear operator arithmetic."""
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse

from probnum import config, utils
from probnum.typing import NotImplementedType, ScalarLike, ShapeLike

from ._arithmetic_fallbacks import (
    NegatedLinearOperator,
    ProductLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    _matmul_fallback,
    _mul_fallback,
)
from ._kronecker import IdentityKronecker, Kronecker, SymmetricKronecker, Symmetrize
from ._linear_operator import (
    BinaryOperandType,
    Embedding,
    Identity,
    LinearOperator,
    Matrix,
    Selection,
    TransposedLinearOperator,
    _InverseLinearOperator,
    _TypeCastLinearOperator,
)
from ._scaling import Scaling, Zero

_AnyLinOp = [
    NegatedLinearOperator,
    ProductLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    Identity,
    IdentityKronecker,
    Matrix,
    TransposedLinearOperator,
    SymmetricKronecker,
    Symmetrize,
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
def _mul_scalar_scaling(scalar: ScalarLike, scaling: Scaling) -> Scaling:
    if scaling.is_isotropic:
        return Scaling(scalar * scaling.scalar, shape=scaling.shape)

    return Scaling(scalar * scaling.factors, shape=scaling.shape)


def _mul_scaling_scalar(scaling: Scaling, scalar: ScalarLike) -> Scaling:
    if scaling.is_isotropic:
        return Scaling(scalar * scaling.scalar, shape=scaling.shape)

    return Scaling(scalar * scaling.factors, shape=scaling.shape)


_mul_fns[(np.number, Scaling)] = _mul_scalar_scaling
_mul_fns[(Scaling, np.number)] = _mul_scaling_scalar
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
    if scaling.shape[1] != kronecker.shape[0]:
        raise ValueError(
            f"matmul received invalid shapes {scaling.shape} @ {kronecker.shape}"
        )

    if scaling.is_isotropic:
        return scaling.scalar * kronecker
    return NotImplemented


def _matmul_kronecker_scaling(kronecker: Kronecker, scaling: Scaling) -> Kronecker:
    if kronecker.shape[1] != scaling.shape[0]:
        raise ValueError(
            f"matmul received invalid shapes {kronecker.shape} @ {scaling.shape}"
        )

    if scaling.is_isotropic:
        return kronecker * scaling.scalar
    return NotImplemented


def _mul_scalar_kronecker(scalar: ScalarLike, kronecker: Kronecker) -> Kronecker:
    if scalar < 0.0:
        return NotImplemented
    sqrt_scalar = np.sqrt(scalar)
    return Kronecker(A=sqrt_scalar * kronecker.A, B=sqrt_scalar * kronecker.B)


def _mul_kronecker_scalar(kronecker: Kronecker, scalar: ScalarLike) -> Kronecker:
    if scalar < 0.0:
        return NotImplemented
    sqrt_scalar = np.sqrt(scalar)
    return Kronecker(A=sqrt_scalar * kronecker.A, B=sqrt_scalar * kronecker.B)


_matmul_fns[(Kronecker, Kronecker)] = Kronecker._matmul_kronecker
_add_fns[(Kronecker, Kronecker)] = Kronecker._add_kronecker
_sub_fns[(Kronecker, Kronecker)] = Kronecker._sub_kronecker

_mul_fns[(np.number, Kronecker)] = _mul_scalar_kronecker
_mul_fns[(Kronecker, np.number)] = _mul_kronecker_scalar
_matmul_fns[(Kronecker, Scaling)] = _matmul_kronecker_scaling
_matmul_fns[(Scaling, Kronecker)] = _matmul_scaling_kronecker


# IdentityKronecker


def _matmul_scaling_idkronecker(
    scaling: Scaling, idkronecker: IdentityKronecker
) -> IdentityKronecker:

    if scaling.shape[1] != idkronecker.shape[0]:
        raise ValueError(
            f"matmul received invalid shapes {scaling.shape} @ {idkronecker.shape}"
        )

    if scaling.is_isotropic:
        return scaling.scalar * idkronecker
    return NotImplemented


def _matmul_idkronecker_scaling(
    idkronecker: IdentityKronecker, scaling: Scaling
) -> IdentityKronecker:

    if idkronecker.shape[1] != scaling.shape[0]:
        raise ValueError(
            f"matmul received invalid shapes {idkronecker.shape} @ {scaling.shape}"
        )

    if scaling.is_isotropic:
        return idkronecker * scaling.scalar
    return NotImplemented


def _mul_scalar_idkronecker(
    scalar: ScalarLike, idkronecker: IdentityKronecker
) -> IdentityKronecker:

    return IdentityKronecker(
        num_blocks=idkronecker.num_blocks, B=scalar * idkronecker.B
    )


def _mul_idkronecker_scalar(
    idkronecker: IdentityKronecker, scalar: ScalarLike
) -> IdentityKronecker:

    return IdentityKronecker(
        num_blocks=idkronecker.num_blocks, B=idkronecker.B * scalar
    )


_matmul_fns[
    (IdentityKronecker, IdentityKronecker)
] = IdentityKronecker._matmul_idkronecker
_add_fns[(IdentityKronecker, IdentityKronecker)] = IdentityKronecker._add_idkronecker
_sub_fns[(IdentityKronecker, IdentityKronecker)] = IdentityKronecker._sub_idkronecker

_mul_fns[(np.number, IdentityKronecker)] = _mul_scalar_idkronecker
_mul_fns[(IdentityKronecker, np.number)] = _mul_idkronecker_scalar
_matmul_fns[(IdentityKronecker, Scaling)] = _matmul_idkronecker_scaling
_matmul_fns[(Scaling, IdentityKronecker)] = _matmul_scaling_idkronecker

_matmul_fns[(Kronecker, IdentityKronecker)] = Kronecker._matmul_kronecker
_matmul_fns[(IdentityKronecker, Kronecker)] = Kronecker._matmul_kronecker

# Matrix
def _matmul_scaling_matrix(scaling: Scaling, matrix: Matrix) -> Matrix:
    return Matrix(A=np.multiply(scaling.factors[:, np.newaxis], matrix.A))


def _matmul_matrix_scaling(matrix: Matrix, scaling: Scaling) -> Matrix:
    return Matrix(A=np.multiply(matrix.A, scaling.factors))


_mul_fns[(Matrix, np.number)] = lambda mat, scal: Matrix(A=scal * mat.A)
_mul_fns[(np.number, Matrix)] = lambda scal, mat: Matrix(A=scal * mat.A)
_matmul_fns[(Matrix, Matrix)] = Matrix._matmul_matrix
_matmul_fns[(Scaling, Matrix)] = _matmul_scaling_matrix
_matmul_fns[(Matrix, Scaling)] = _matmul_matrix_scaling


_matmul_fns[(Selection, Matrix)] = lambda sel, mat: Matrix(sel @ mat.A)
_matmul_fns[(Embedding, Matrix)] = lambda emb, mat: Matrix(emb @ mat.A)
_matmul_fns[(Matrix, Selection)] = lambda mat, sel: Matrix(mat.A @ sel)
_matmul_fns[(Matrix, Embedding)] = lambda mat, emb: Matrix(mat.A @ emb)

_add_fns[(Matrix, Matrix)] = lambda mat1, mat2: Matrix(mat1.A + mat2.A)
_sub_fns[(Matrix, Matrix)] = lambda mat1, mat2: Matrix(mat1.A - mat2.A)


def _matmul_matrix_wrapped(
    mat: Matrix, wrapped: Union[_InverseLinearOperator, TransposedLinearOperator]
) -> Union[Matrix, NotImplementedType]:
    if not config.lazy_matrix_matrix_matmul:
        return Matrix(mat.A @ wrapped)
    return NotImplemented


def _matmul_wrapped_matrix(
    wrapped: Union[_InverseLinearOperator, TransposedLinearOperator], mat: Matrix
) -> Union[Matrix, NotImplementedType]:
    if not config.lazy_matrix_matrix_matmul:
        return Matrix(wrapped @ mat.A)
    return NotImplemented


_matmul_fns[(Matrix, _InverseLinearOperator)] = _matmul_matrix_wrapped
_matmul_fns[(_InverseLinearOperator, Matrix)] = _matmul_wrapped_matrix
_matmul_fns[(Matrix, TransposedLinearOperator)] = _matmul_matrix_wrapped
_matmul_fns[(TransposedLinearOperator, Matrix)] = _matmul_wrapped_matrix


# Identity
def _matmul_id_any(idty: Identity, anyop: LinearOperator) -> LinearOperator:
    if idty.shape[1] != anyop.shape[0]:
        raise ValueError(f"matmul received invalid shapes {idty.shape} @ {anyop.shape}")

    return anyop


def _matmul_any_id(anyop: LinearOperator, idty: Identity) -> LinearOperator:
    if anyop.shape[1] != idty.shape[0]:
        raise ValueError(f"matmul received invalid shapes {anyop.shape} @ {idty.shape}")

    return anyop


for op_type in _AnyLinOp:
    _matmul_fns[(Identity, op_type)] = _matmul_id_any
    _matmul_fns[(op_type, Identity)] = _matmul_any_id


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
        raise ValueError(f"matmul received invalid shapes {z.shape} @ {op.shape}")

    return Zero(shape=(z.shape[0], op.shape[1]))


def _matmul_anylinop_zero(op: LinearOperator, z: Zero) -> Zero:
    if op.shape[1] != z.shape[0]:
        raise ValueError(f"matmul received invalid shapes {op.shape} @ {z.shape}")

    return Zero(shape=(op.shape[0], z.shape[1]))


def _add_zero_anylinop(z: Zero, op: LinearOperator) -> Zero:
    if z.shape != op.shape:
        raise ValueError(f"add received invalid shapes {z.shape} + {op.shape}")

    return op


def _add_anylinop_zero(op: LinearOperator, z: Zero) -> Zero:
    if z.shape != op.shape:
        raise ValueError(f"add received invalid shapes {op.shape} + {z.shape}")

    return op


def _sub_zero_anylinop(z: Zero, op: LinearOperator) -> Zero:
    if z.shape != op.shape:
        raise ValueError(f"sub received invalid shapes {op.shape} - {z.shape}")

    return -op


def _sub_anylinop_zero(op: LinearOperator, z: Zero) -> Zero:
    if z.shape != op.shape:
        raise ValueError(f"sub received invalid shapes {op.shape} - {z.shape}")

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
    if np.ndim(op1) == 0:
        key1 = np.number
        op1 = utils.as_numpy_scalar(op1)
    else:
        key1 = type(op1)

    if np.ndim(op2) == 0:
        key2 = np.number
        op2 = utils.as_numpy_scalar(op2)
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


def _operand_to_linop(operand: Any, shape: ShapeLike) -> Optional[LinearOperator]:
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
