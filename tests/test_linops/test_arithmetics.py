"""Tests for linear operator arithmetics."""

import itertools
import operator

import numpy as np
import pytest

from probnum.linops._arithmetic import _add_fns, _matmul_fns, _mul_fns, _sub_fns
from probnum.linops._arithmetic_fallbacks import (
    NegatedLinearOperator,
    ProductLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
)
from probnum.linops._kronecker import (
    IdentityKronecker,
    Kronecker,
    SymmetricKronecker,
    Symmetrize,
)
from probnum.linops._linear_operator import (
    AdjointLinearOperator,
    Embedding,
    Identity,
    Matrix,
    Selection,
    TransposedLinearOperator,
    _ConjugateLinearOperator,
    _InverseLinearOperator,
    _TypeCastLinearOperator,
)
from probnum.linops._scaling import Scaling, Zero
from probnum.problems.zoo.linalg import random_spd_matrix


def _aslist(arg):
    try:
        return list(arg)
    except TypeError:
        return [arg]


def get_linop(linop_type):
    # pylint: disable=too-many-return-statements
    if linop_type is Kronecker:
        return Kronecker(np.random.rand(2, 2), np.random.rand(2, 2))
    elif linop_type is IdentityKronecker:
        return IdentityKronecker(2, np.random.rand(2, 2))
    elif linop_type is Zero or linop_type is Identity:
        return linop_type(shape=(4, 4))
    elif linop_type is Scaling:
        return (
            Scaling(factors=np.random.rand(4)),
            Scaling(factors=3.14 * np.ones(4)),
        )
    elif linop_type is Matrix:
        return Matrix(np.random.rand(4, 4))
    elif linop_type is _InverseLinearOperator:
        _posdef_randmat = random_spd_matrix(rng=np.random.default_rng(123), dim=4)
        return Matrix(_posdef_randmat).inv()
    elif linop_type is TransposedLinearOperator:
        return TransposedLinearOperator(linop=Matrix(np.random.rand(4, 4)))
    elif linop_type is Embedding:
        return Embedding(take_indices=(0, 1, 2), put_indices=(1, 0, 3), shape=(4, 3))
    elif linop_type is Selection:
        return Selection(indices=(1, 0, 3), shape=(3, 4))
    elif linop_type is NegatedLinearOperator:
        return NegatedLinearOperator(linop=Matrix(np.random.rand(4, 4)))
    elif linop_type is ScaledLinearOperator:
        return ScaledLinearOperator(linop=Matrix(np.random.rand(4, 4)), scalar=3.14)
    elif linop_type is ProductLinearOperator:
        return ProductLinearOperator(
            Matrix(np.random.rand(4, 4)), Matrix(np.random.rand(4, 4))
        )
    elif linop_type is SumLinearOperator:
        return SumLinearOperator(
            Matrix(np.random.rand(4, 4)), Matrix(np.random.rand(4, 4))
        )
    elif linop_type is AdjointLinearOperator:
        return AdjointLinearOperator(linop=Identity(4))
    elif linop_type is _ConjugateLinearOperator:
        return _ConjugateLinearOperator(linop=Identity(4, dtype=np.complex64))
    elif linop_type is SymmetricKronecker:
        return SymmetricKronecker(Identity(2), Identity(2))
    elif linop_type is Symmetrize:
        return Symmetrize(2)
    elif linop_type is _TypeCastLinearOperator:
        return _TypeCastLinearOperator(
            linop=Matrix(np.random.rand(4, 4)), dtype=np.float32
        )
    elif isinstance(linop_type, str) and linop_type == "scalar":
        return 1.3579
    else:
        raise TypeError(f"Don't know what to do with type {linop_type}.")


@pytest.mark.parametrize(
    "op", [operator.matmul, operator.mul, operator.add, operator.sub]
)
def test_arithmetics(op):

    registry = {
        operator.matmul: _matmul_fns,
        operator.mul: _mul_fns,
        operator.add: _add_fns,
        operator.sub: _sub_fns,
    }[op]

    for (l_type, r_type) in registry.keys():
        if (
            l_type is Selection
            or l_type is Embedding
            or r_type is Selection
            or r_type is Embedding
        ):
            # Checked seperatly
            continue

        linops1 = get_linop(l_type)
        linops2 = get_linop(r_type)

        for (linop1, linop2) in itertools.product(_aslist(linops1), _aslist(linops2)):

            res_linop = op(linop1, linop2)
            assert res_linop.ndim == 2

            if isinstance(l_type, str) and l_type == "scalar":
                assert res_linop.shape == linop2.shape
            elif isinstance(r_type, str) and r_type == "scalar":
                assert res_linop.shape == linop1.shape
            else:
                assert res_linop.shape[0] == linop1.shape[0]
                assert res_linop.shape[1] == linop2.shape[1]
