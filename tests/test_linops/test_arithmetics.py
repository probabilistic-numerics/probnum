"""Tests for linear operator arithmetics."""
# pylint: disable=consider-iterating-dictionary

import itertools

import numpy as np
import pytest

from probnum import config
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
    """Converts anything to a list. Non-iterables become single-element lists."""
    try:
        return list(arg)
    except TypeError:  # excepts TypeError: '<type>' object is not iterable
        return [arg]


def get_linop(linop_type):
    # pylint: disable=too-many-return-statements
    if linop_type is Kronecker:
        _A1 = np.random.rand(2, 2)
        _B1 = np.random.rand(2, 3)
        return (
            Kronecker(_A1, np.random.rand(2, 2)),
            Kronecker(np.random.rand(4, 3), _B1),
            Kronecker(_A1, np.random.rand(2, 2)),
            Kronecker(np.random.rand(2, 2), _B1),
        )
    elif linop_type is IdentityKronecker:
        return (
            IdentityKronecker(2, np.random.rand(2, 2)),
            IdentityKronecker(3, np.random.rand(3, 4)),
        )
    elif linop_type is Zero or linop_type is Identity:
        return (linop_type(shape=(4, 4)), linop_type(shape=(3, 3)))
    elif linop_type is Scaling:
        return (
            Scaling(factors=np.random.rand(4)),
            Scaling(factors=3.14, shape=(4, 4)),
            Scaling(factors=np.random.rand(6), shape=(6, 6)),
            Scaling(factors=3.14, shape=(3, 3)),
        )
    elif linop_type is Matrix:
        return (Matrix(np.random.rand(4, 4)), Matrix(np.random.rand(6, 3)))
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


def test_matmul():

    for (l_type, r_type) in _matmul_fns.keys():
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

            if linop1.shape[1] != linop2.shape[0]:
                with pytest.raises(ValueError):
                    res_linop = linop1 @ linop2
            else:
                res_linop = linop1 @ linop2
                assert res_linop.ndim == 2

                if isinstance(l_type, str) and l_type == "scalar":
                    assert res_linop.shape == linop2.shape
                elif isinstance(r_type, str) and r_type == "scalar":
                    assert res_linop.shape == linop1.shape
                else:
                    assert res_linop.shape[0] == linop1.shape[0]
                    assert res_linop.shape[1] == linop2.shape[1]


def test_mul():

    for (l_type, r_type) in _mul_fns.keys():
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

            if isinstance(l_type, str) and l_type == "scalar":
                res_linop = linop1 * linop2
                assert res_linop.shape == linop2.shape
            elif isinstance(r_type, str) and r_type == "scalar":
                res_linop = linop1 * linop2
                assert res_linop.shape == linop1.shape
            else:
                if linop1.shape != linop2.shape:
                    with pytest.raises(ValueError):
                        res_linop = linop1 * linop2
                else:
                    res_linop = linop1 * linop2
                    assert res_linop.shape == linop1.shape == linop2.shape


def test_add():

    for (l_type, r_type) in _add_fns.keys():
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

            if linop1.shape != linop2.shape:
                with pytest.raises(ValueError):
                    res_linop = linop1 + linop2
            else:
                res_linop = linop1 + linop2
                assert res_linop.shape == linop1.shape == linop2.shape


def test_sub():

    for (l_type, r_type) in _sub_fns.keys():
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

            if linop1.shape != linop2.shape:
                with pytest.raises(ValueError):
                    res_linop = linop1 - linop2
            else:
                res_linop = linop1 - linop2
                assert res_linop.shape == linop1.shape == linop2.shape


def test_kronecker_matmul():
    # Checks the case in which the shapes of the Kronecker-structured matrices
    # are valid in itself but the respective Kronecker factors (k1.A @ k2.A and/or
    # k1.B @ k2.B) have invalid shapes for matmul.
    k1 = Kronecker(np.random.rand(4, 2), np.random.rand(2, 3))  # (8, 6)
    k2 = Kronecker(np.random.rand(3, 2), np.random.rand(2, 3))  # (6, 6)

    # Even though the shapes fit, and Kronecker @ Kronecker = Kronecker ....
    assert k1.shape[1] == k2.shape[0]

    # The result does not have a Kronecker structure
    res = k1 @ k2
    assert not isinstance(res, Kronecker)


def test_selection_embedding():
    sel = get_linop(Selection)
    emb = get_linop(Embedding)
    emb2 = Embedding(
        take_indices=emb._take_indices, put_indices=emb._put_indices, shape=(5, 3)
    )

    product1 = sel @ emb
    assert product1.shape[0] == sel.shape[0]
    assert product1.shape[1] == emb.shape[1]

    product2 = sel @ emb2
    assert product2.shape[0] == sel.shape[0]
    assert product2.shape[1] == emb2.shape[1]


def test_lazy_matrix_matrix_matmul_option():
    mat1 = get_linop(Matrix)[0]
    mat2 = get_linop(Matrix)[0]
    inv = get_linop(_InverseLinearOperator)
    transposed = get_linop(TransposedLinearOperator)

    with config(lazy_matrix_matrix_matmul=False):
        assert isinstance(mat1 @ mat2, ProductLinearOperator)
        assert isinstance(mat1 @ inv, ProductLinearOperator)
        assert isinstance(inv @ mat2, ProductLinearOperator)
        assert isinstance(mat1 @ transposed, ProductLinearOperator)
        assert isinstance(transposed @ mat2, ProductLinearOperator)

    with config(lazy_matrix_matrix_matmul=True):
        assert isinstance(mat1 @ mat2, Matrix)
        assert isinstance(mat1 @ inv, Matrix)
        assert isinstance(inv @ mat2, Matrix)
        assert isinstance(mat1 @ transposed, Matrix)
        assert isinstance(transposed @ mat2, Matrix)


def test_equality():
    scalings = get_linop(Scaling)
    int_scaling = Scaling(2, shape=(4, 4))
    for s1, s2 in itertools.product(scalings, _aslist(scalings) + [int_scaling]):
        if (
            s1.shape == s2.shape
            and s1.dtype == s2.dtype
            and np.all(s1.todense() == s2.todense())
        ):
            assert s1 == s2

        else:
            assert s1 != s2
