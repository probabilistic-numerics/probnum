import operator

from probnum.prob._random_variable import RandomVariable
from probnum.prob.random_variable import Dirac, Normal


def add(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_add_fns, rv1, rv2)


def sub(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_sub_fns, rv1, rv2)


def mul(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_mul_fns, rv1, rv2)


def matmul(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_matmul_fns, rv1, rv2)


def truediv(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_truediv_fns, rv1, rv2)


def floordiv(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_floordiv_fns, rv1, rv2)


def mod(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_mod_fns, rv1, rv2)


def divmod_(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_divmod_fns, rv1, rv2)


def pow_(rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
    return _apply(_pow_fns, rv1, rv2)


# Operator registry
def _apply(op_registry, rv1: RandomVariable, rv2: RandomVariable) -> RandomVariable:
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
_add_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(operator.add)
_sub_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(operator.sub)
_mul_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(operator.mul)
_matmul_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(operator.matmul)
_truediv_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(operator.truediv)
_floordiv_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(operator.floordiv)
_mod_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(operator.mod)
_divmod_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(divmod)
_pow_fns[(Dirac, Dirac)] = Dirac._binary_operator_factory(operator.pow)

# Normal
_add_fns[(Normal, Normal)] = Normal._add_normal
_add_fns[(Normal, Dirac)] = Normal._add_dirac
_add_fns[(Dirac, Normal)] = _swap_operands(Normal._add_dirac)

_sub_fns[(Normal, Normal)] = Normal._sub_normal
_sub_fns[(Normal, Dirac)] = Normal._sub_dirac
_sub_fns[(Dirac, Normal)] = _swap_operands(Normal._rsub_dirac)

_mul_fns[(Normal, Dirac)] = Normal._mul_dirac
_mul_fns[(Dirac, Normal)] = _swap_operands(Normal._mul_dirac)

_matmul_fns[(Normal, Dirac)] = Normal._matmul_dirac
_matmul_fns[(Dirac, Normal)] = _swap_operands(Normal._rmatmul_dirac)

_truediv_fns[(Normal, Dirac)] = Normal._truediv_dirac
