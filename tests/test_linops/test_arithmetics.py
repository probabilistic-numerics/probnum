"""Tests for linear operator arithmetics."""
import numpy as np
import pytest

from probnum import linops

matrix_shape = (4, 4)

kron_factor_shape = (2, 2)


def get_kronecker():
    return linops.Kronecker(
        np.random.rand(*kron_factor_shape), np.random.rand(*kron_factor_shape)
    )


def get_zero():
    return linops.Zero(matrix_shape)


def get_scaling():
    return linops.Scaling(factors=np.random.rand(matrix_shape[0]))


def get_inverse_linop():
    rand_matrix = np.random.rand(*matrix_shape)
    _mat = linops.Matrix(rand_matrix.T @ rand_matrix)
    return _mat.inv()


def get_matrix_linop():
    return linops.Matrix(np.random.rand(*matrix_shape))


def get_idkron_linop():
    return linops.IdentityKronecker(
        num_blocks=kron_factor_shape[0], B=np.random.rand(*kron_factor_shape)
    )


def get_id_linop():
    return linops.Identity(matrix_shape[0])


def get_sum_linop():
    summand1 = linops.Matrix(np.random.rand(*matrix_shape))
    summand2 = linops.Matrix(np.random.rand(*matrix_shape))
    return summand1 + summand2


def get_product_linop():
    factor1 = linops.Matrix(np.random.rand(*matrix_shape))
    factor2 = linops.Matrix(np.random.rand(*matrix_shape))
    return factor1 @ factor2


def get_negated_linop():
    op = linops.Matrix(np.random.rand(*matrix_shape))
    return -op


def get_scaled_linop():
    op = linops.Matrix(np.random.rand(*matrix_shape))
    return 3.14 * op


def get_transposed_linop():
    op = linops.Matrix(np.random.rand(*matrix_shape))
    return op.T


@pytest.mark.parametrize(
    "linop1",
    [
        get_kronecker(),
        get_zero(),
        get_scaling(),
        get_inverse_linop(),
        get_matrix_linop(),
        get_idkron_linop(),
        get_id_linop(),
        get_sum_linop(),
        get_product_linop(),
        get_negated_linop(),
        get_scaled_linop(),
        get_transposed_linop(),
    ],
)
@pytest.mark.parametrize(
    "linop2",
    [
        get_kronecker(),
        get_zero(),
        get_scaling(),
        get_inverse_linop(),
        get_matrix_linop(),
        get_idkron_linop(),
        get_id_linop(),
        get_sum_linop(),
        get_product_linop(),
        get_negated_linop(),
        get_scaled_linop(),
        get_transposed_linop(),
    ],
)
def test_arithmetics(linop1, linop2):

    res_linop = linop1 @ linop2
    assert res_linop.ndim == 2
    assert res_linop.shape[0] == linop1.shape[0]
    assert res_linop.shape[1] == linop2.shape[1]
