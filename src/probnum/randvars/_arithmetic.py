"""This module implements binary arithmetic operators between pairs of random
variables."""

import operator
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

import probnum.linops as _linear_operators
from probnum import utils as _utils

from ._constant import Constant as _Constant
from ._normal import Normal as _Normal
from ._random_variable import RandomVariable as _RandomVariable
from ._utils import asrandvar as _asrandvar


def add(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_add_fns, rv1, rv2)


def sub(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_sub_fns, rv1, rv2)


def mul(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_mul_fns, rv1, rv2)


def matmul(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_matmul_fns, rv1, rv2)


def truediv(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_truediv_fns, rv1, rv2)


def floordiv(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_floordiv_fns, rv1, rv2)


def mod(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_mod_fns, rv1, rv2)


def divmod_(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_divmod_fns, rv1, rv2)


def pow_(rv1: Any, rv2: Any) -> _RandomVariable:
    return _apply(_pow_fns, rv1, rv2)


########################################################################################
# Operator registry
########################################################################################

_RandomVariableBinaryOperator = Callable[
    [_RandomVariable, _RandomVariable], Union[_RandomVariable, type(NotImplemented)]
]
_OperatorRegistryType = Dict[Tuple[type, type], _RandomVariableBinaryOperator]


_add_fns: _OperatorRegistryType = {}
_sub_fns: _OperatorRegistryType = {}
_mul_fns: _OperatorRegistryType = {}
_matmul_fns: _OperatorRegistryType = {}
_truediv_fns: _OperatorRegistryType = {}
_floordiv_fns: _OperatorRegistryType = {}
_mod_fns: _OperatorRegistryType = {}
_divmod_fns: _OperatorRegistryType = {}
_pow_fns: _OperatorRegistryType = {}


def _apply(
    op_registry: _OperatorRegistryType,
    rv1: Any,
    rv2: Any,
) -> Union[_RandomVariable, type(NotImplemented)]:
    # Convert arguments to random variables
    rv1 = _asrandvar(rv1)
    rv2 = _asrandvar(rv2)

    # Search specific operatir
    key = (type(rv1), type(rv2))

    if key in op_registry:
        res = op_registry[key](rv1, rv2)
    else:
        res = NotImplemented

    if res is NotImplemented:
        res = op_registry[(_RandomVariable, _RandomVariable)](rv1, rv2)

    return res


####################
# Helper Functions #
####################


def _swap_operands(fn: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    return lambda op1, op2: fn(op2, op1)


########################################################################################
# Generic Random Variable Arithmetic (Fallbacks)
########################################################################################


def _default_rv_binary_op_factory(op_fn) -> _RandomVariableBinaryOperator:
    def _rv_binary_op(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
        shape, dtype, sample = _make_rv_binary_op_result_shape_dtype_sample_fn(
            op_fn, rv1, rv2
        )

        return _RandomVariable(
            shape=shape,
            dtype=dtype,
            sample=sample,
        )

    return _rv_binary_op


def _make_rv_binary_op_result_shape_dtype_sample_fn(op_fn, rv1, rv2):
    sample_fn = lambda size: op_fn(rv1.sample(size), rv2.sample(size))

    # Infer shape and dtype
    infer_sample = sample_fn(())

    shape = infer_sample.shape
    dtype = infer_sample.dtype

    return shape, dtype, sample_fn


def _generic_rv_add(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    shape, dtype, sample = _make_rv_binary_op_result_shape_dtype_sample_fn(
        operator.add, rv1, rv2
    )

    return _RandomVariable(
        shape=shape,
        dtype=dtype,
        sample=sample,
        mean=lambda: rv1.mean + rv2.mean,
    )


_add_fns[(_RandomVariable, _RandomVariable)] = _generic_rv_add
_sub_fns[(_RandomVariable, _RandomVariable)] = _default_rv_binary_op_factory(
    operator.sub
)
_mul_fns[(_RandomVariable, _RandomVariable)] = _default_rv_binary_op_factory(
    operator.mul
)
_matmul_fns[(_RandomVariable, _RandomVariable)] = _default_rv_binary_op_factory(
    operator.matmul
)
_truediv_fns[(_RandomVariable, _RandomVariable)] = _default_rv_binary_op_factory(
    operator.truediv
)
_floordiv_fns[(_RandomVariable, _RandomVariable)] = _default_rv_binary_op_factory(
    operator.floordiv
)
_mod_fns[(_RandomVariable, _RandomVariable)] = _default_rv_binary_op_factory(
    operator.mod
)
_divmod_fns[(_RandomVariable, _RandomVariable)] = _default_rv_binary_op_factory(divmod)
_pow_fns[(_RandomVariable, _RandomVariable)] = _default_rv_binary_op_factory(
    operator.pow
)


########################################################################################
# Constant - Constant Arithmetic
########################################################################################

_add_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(operator.add)
_sub_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(operator.sub)
_mul_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(operator.mul)
_matmul_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(
    operator.matmul
)
_truediv_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(
    operator.truediv
)
_floordiv_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(
    operator.floordiv
)
_mod_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(operator.mod)
_divmod_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(divmod)
_pow_fns[(_Constant, _Constant)] = _Constant._binary_operator_factory(operator.pow)

########################################################################################
# Normal - Normal Arithmetic
########################################################################################

_add_fns[(_Normal, _Normal)] = _Normal._add_normal
_sub_fns[(_Normal, _Normal)] = _Normal._sub_normal


########################################################################################
# Normal - Constant Arithmetic
########################################################################################


def _add_normal_constant(norm_rv: _Normal, constant_rv: _Constant) -> _Normal:
    cov_cholesky = norm_rv.cov_cholesky if norm_rv.cov_cholesky_is_precomputed else None
    return _Normal(
        mean=norm_rv.mean + constant_rv.support,
        cov=norm_rv.cov,
        cov_cholesky=cov_cholesky,
    )


_add_fns[(_Normal, _Constant)] = _add_normal_constant
_add_fns[(_Constant, _Normal)] = _swap_operands(_add_normal_constant)


def _sub_normal_constant(norm_rv: _Normal, constant_rv: _Constant) -> _Normal:
    cov_cholesky = norm_rv.cov_cholesky if norm_rv.cov_cholesky_is_precomputed else None
    return _Normal(
        mean=norm_rv.mean - constant_rv.support,
        cov=norm_rv.cov,
        cov_cholesky=cov_cholesky,
    )


_sub_fns[(_Normal, _Constant)] = _sub_normal_constant


def _sub_constant_normal(constant_rv: _Constant, norm_rv: _Normal) -> _Normal:
    cov_cholesky = norm_rv.cov_cholesky if norm_rv.cov_cholesky_is_precomputed else None
    return _Normal(
        mean=constant_rv.support - norm_rv.mean,
        cov=norm_rv.cov,
        cov_cholesky=cov_cholesky,
    )


_sub_fns[(_Constant, _Normal)] = _sub_constant_normal


def _mul_normal_constant(
    norm_rv: _Normal, constant_rv: _Constant
) -> Union[_Normal, _Constant, type(NotImplemented)]:
    if constant_rv.size == 1:
        if constant_rv.support == 0:
            return _Constant(
                support=np.zeros_like(norm_rv.mean),
            )
        else:
            if norm_rv.cov_cholesky_is_precomputed:
                cov_cholesky = constant_rv.support * norm_rv.cov_cholesky
            else:
                cov_cholesky = None
            return _Normal(
                mean=constant_rv.support * norm_rv.mean,
                cov=(constant_rv.support ** 2) * norm_rv.cov,
                cov_cholesky=cov_cholesky,
            )

    return NotImplemented


_mul_fns[(_Normal, _Constant)] = _mul_normal_constant
_mul_fns[(_Constant, _Normal)] = _swap_operands(_mul_normal_constant)


def _matmul_normal_constant(norm_rv: _Normal, constant_rv: _Constant) -> _Normal:
    if norm_rv.ndim == 1 or (norm_rv.ndim == 2 and norm_rv.shape[0] == 1):
        if norm_rv.cov_cholesky_is_precomputed:
            cov_cholesky = _utils.linalg.cholesky_update(
                constant_rv.support.T @ norm_rv.cov_cholesky
            )
        else:
            cov_cholesky = None
        return _Normal(
            mean=norm_rv.mean @ constant_rv.support,
            cov=constant_rv.support.T @ (norm_rv.cov @ constant_rv.support),
            cov_cholesky=cov_cholesky,
        )
    elif norm_rv.ndim == 2 and norm_rv.shape[0] > 1:
        # This part does not do the Cholesky update,
        # because of performance configurations: currently, there is no way of switching
        # the Cholesky updates off, which might affect (large, potentially sparse) covariance matrices
        # of matrix-variate Normal RVs. See Issue #335.
        cov_update = _linear_operators.Kronecker(
            _linear_operators.Identity(constant_rv.shape[0]), constant_rv.support
        )

        return _Normal(
            mean=norm_rv.mean @ constant_rv.support,
            cov=cov_update.T @ (norm_rv.cov @ cov_update),
        )
    else:
        raise TypeError(
            "Currently, matrix multiplication is only supported for vector- and "
            "matrix-variate Gaussians."
        )


_matmul_fns[(_Normal, _Constant)] = _matmul_normal_constant


def _matmul_constant_normal(constant_rv: _Constant, norm_rv: _Normal) -> _Normal:
    if norm_rv.ndim == 1 or (norm_rv.ndim == 2 and norm_rv.shape[1] == 1):
        if norm_rv.cov_cholesky_is_precomputed:
            cov_cholesky = _utils.linalg.cholesky_update(
                constant_rv.support @ norm_rv.cov_cholesky
            )
        else:
            cov_cholesky = None
        return _Normal(
            mean=constant_rv.support @ norm_rv.mean,
            cov=constant_rv.support @ (norm_rv.cov @ constant_rv.support.T),
            cov_cholesky=cov_cholesky,
        )
    else:
        raise TypeError(
            "Currently, matrix multiplication is only supported for vector-variate "
            "Gaussians."
        )


_matmul_fns[(_Constant, _Normal)] = _matmul_constant_normal


def _truediv_normal_constant(norm_rv: _Normal, constant_rv: _Constant) -> _Normal:
    if constant_rv.size == 1:
        if constant_rv.support == 0:
            raise ZeroDivisionError

        if norm_rv.cov_cholesky_is_precomputed:
            cov_cholesky = norm_rv.cov_cholesky / constant_rv.support
        else:
            cov_cholesky = None

        return _Normal(
            mean=norm_rv.mean / constant_rv.support,
            cov=norm_rv.cov / (constant_rv.support ** 2),
            cov_cholesky=cov_cholesky,
        )

    return NotImplemented


_truediv_fns[(_Normal, _Constant)] = _truediv_normal_constant
