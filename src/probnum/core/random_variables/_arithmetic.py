import operator

from ._random_variable import RandomVariable as _RandomVariable
from probnum.core.random_variables import Dirac as _Dirac, Normal as _Normal


def add(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_add_fns, rv1, rv2)


def sub(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_sub_fns, rv1, rv2)


def mul(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_mul_fns, rv1, rv2)


def matmul(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_matmul_fns, rv1, rv2)


def truediv(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_truediv_fns, rv1, rv2)


def floordiv(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_floordiv_fns, rv1, rv2)


def mod(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_mod_fns, rv1, rv2)


def divmod_(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_divmod_fns, rv1, rv2)


def pow_(rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    return _apply(_pow_fns, rv1, rv2)


# Operator registry
def _apply(op_registry, rv1: _RandomVariable, rv2: _RandomVariable) -> _RandomVariable:
    key = (type(rv1), type(rv2))

    if key not in op_registry:
        return NotImplemented

    res = op_registry[key](rv1, rv2)

    return res


_add_fns = {}
_sub_fns = {}
_mul_fns = {}
_matmul_fns = {}
_truediv_fns = {}
_floordiv_fns = {}
_mod_fns = {}
_divmod_fns = {}
_pow_fns = {}


def _swap_operands(fn):
    return lambda op1, op2: fn(op2, op1)


# Dirac
_add_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.add)
_sub_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.sub)
_mul_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.mul)
_matmul_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.matmul)
_truediv_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.truediv)
_floordiv_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.floordiv)
_mod_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.mod)
_divmod_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(divmod)
_pow_fns[(_Dirac, _Dirac)] = _Dirac._binary_operator_factory(operator.pow)

# Normal
_add_fns[(_Normal, _Normal)] = _Normal._add_normal
_add_fns[(_Normal, _Dirac)] = _Normal._add_dirac
_add_fns[(_Dirac, _Normal)] = _swap_operands(_Normal._add_dirac)

_sub_fns[(_Normal, _Normal)] = _Normal._sub_normal
_sub_fns[(_Normal, _Dirac)] = _Normal._sub_dirac
_sub_fns[(_Dirac, _Normal)] = _swap_operands(_Normal._rsub_dirac)

_mul_fns[(_Normal, _Dirac)] = _Normal._mul_dirac
_mul_fns[(_Dirac, _Normal)] = _swap_operands(_Normal._mul_dirac)

_matmul_fns[(_Normal, _Dirac)] = _Normal._matmul_dirac
_matmul_fns[(_Dirac, _Normal)] = _swap_operands(_Normal._rmatmul_dirac)

_truediv_fns[(_Normal, _Dirac)] = _Normal._truediv_dirac
