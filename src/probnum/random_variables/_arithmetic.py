""" This module implements binary arithmetic operators between pairs of random
variables. """

import operator
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

from probnum import utils as _utils
from probnum.linalg import linops as _linops

from ._utils import asrandvar as _asrandvar
from ._random_variable import RandomVariable as _RandomVariable
from ._dirac import Dirac as _Dirac
from ._normal import Normal as _Normal


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
            random_state=_utils.derive_random_seed(rv1.random_state, rv2.random_state),
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
        random_state=_utils.derive_random_seed(rv1.random_state, rv2.random_state),
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
# Dirac - Dirac Arithmetic
########################################################################################

_add_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.add)
_sub_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.sub)
_mul_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.mul)
_matmul_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.matmul)
_truediv_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.truediv)
_floordiv_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.floordiv)
_mod_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.mod)
_divmod_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(divmod)
_pow_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.pow)

########################################################################################
# Normal - Normal Arithmetic
########################################################################################

_add_fns[(_Normal, _Normal)] = _Normal._add_normal
_sub_fns[(_Normal, _Normal)] = _Normal._sub_normal


########################################################################################
# Normal - Dirac Arithmetic
########################################################################################


def _add_normal_dirac(norm_rv: _Normal, dirac_rv: _Dirac) -> _Normal:
    return _Normal(
        mean=norm_rv.mean + dirac_rv.support,
        cov=norm_rv.cov,
        random_state=_utils.derive_random_seed(
            norm_rv.random_state, dirac_rv.random_state
        ),
    )


_add_fns[(_Normal, _Dirac)] = _add_normal_dirac
_add_fns[(_Dirac, _Normal)] = _swap_operands(_add_normal_dirac)


def _sub_normal_dirac(norm_rv: _Normal, dirac_rv: _Dirac) -> _Normal:
    return _Normal(
        mean=norm_rv.mean - dirac_rv.support,
        cov=norm_rv.cov,
        random_state=_utils.derive_random_seed(
            norm_rv.random_state, dirac_rv.random_state
        ),
    )


_sub_fns[(_Normal, _Dirac)] = _sub_normal_dirac


def _sub_dirac_normal(dirac_rv: _Dirac, norm_rv: _Normal) -> _Normal:
    return _Normal(
        mean=dirac_rv.support - norm_rv.mean,
        cov=norm_rv.cov,
        random_state=_utils.derive_random_seed(
            dirac_rv.random_state, norm_rv.random_state
        ),
    )


_sub_fns[(_Dirac, _Normal)] = _sub_dirac_normal


def _mul_normal_dirac(
    norm_rv: _Normal, dirac_rv: _Dirac
) -> Union[_Normal, _Dirac, type(NotImplemented)]:
    if dirac_rv.size == 1:
        if dirac_rv.support == 0:
            return _Dirac(
                support=np.zeros_like(norm_rv.mean),
                random_state=_utils.derive_random_seed(
                    norm_rv.random_state, dirac_rv.random_state
                ),
            )
        else:
            return _Normal(
                mean=dirac_rv.support * norm_rv.mean,
                cov=(dirac_rv.support ** 2) * norm_rv.cov,
                random_state=_utils.derive_random_seed(
                    norm_rv.random_state, dirac_rv.random_state
                ),
            )

    return NotImplemented


_mul_fns[(_Normal, _Dirac)] = _mul_normal_dirac
_mul_fns[(_Dirac, _Normal)] = _swap_operands(_mul_normal_dirac)


def _matmul_normal_dirac(norm_rv: _Normal, dirac_rv: _Dirac) -> _Normal:
    if norm_rv.ndim == 1 or (norm_rv.ndim == 2 and norm_rv.shape[0] == 1):
        return _Normal(
            mean=norm_rv.mean @ dirac_rv.support,
            cov=dirac_rv.support.T @ (norm_rv.cov @ dirac_rv.support),
            random_state=_utils.derive_random_seed(
                norm_rv.random_state, dirac_rv.random_state
            ),
        )
    elif norm_rv.ndim == 2 and norm_rv.shape[0] > 1:
        cov_update = _linops.Kronecker(
            _linops.Identity(dirac_rv.shape[0]), dirac_rv.support
        )

        return _Normal(
            mean=norm_rv.mean @ dirac_rv.support,
            cov=cov_update.T @ (norm_rv.cov @ cov_update),
            random_state=_utils.derive_random_seed(
                norm_rv.random_state, dirac_rv.random_state
            ),
        )
    else:
        raise TypeError(
            "Currently, matrix multiplication is only supported for vector- and "
            "matrix-variate Gaussians."
        )


_matmul_fns[(_Normal, _Dirac)] = _matmul_normal_dirac


def _matmul_dirac_normal(dirac_rv: _Dirac, norm_rv: _Normal) -> _Normal:
    if norm_rv.ndim == 1 or (norm_rv.ndim == 2 and norm_rv.shape[1] == 1):
        return _Normal(
            mean=dirac_rv.support @ norm_rv.mean,
            cov=dirac_rv.support @ (norm_rv.cov @ dirac_rv.support.T),
            random_state=_utils.derive_random_seed(
                dirac_rv.random_state, norm_rv.random_state
            ),
        )
    else:
        raise TypeError(
            "Currently, matrix multiplication is only supported for vector-variate "
            "Gaussians."
        )


_matmul_fns[(_Dirac, _Normal)] = _matmul_dirac_normal


def _truediv_normal_dirac(norm_rv: _Normal, dirac_rv: _Dirac) -> _Normal:
    if dirac_rv.size == 1:
        if dirac_rv.support == 0:
            raise ZeroDivisionError

        return _Normal(
            mean=norm_rv.mean / dirac_rv.support,
            cov=norm_rv.cov / (dirac_rv.support ** 2),
            random_state=_utils.derive_random_seed(
                norm_rv.random_state, dirac_rv.random_state
            ),
        )

    return NotImplemented


_truediv_fns[(_Normal, _Dirac)] = _truediv_normal_dirac
