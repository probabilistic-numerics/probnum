"""Tests for linear operator arithmetics fallbacks."""

import numpy as np

# NegatedLinearOperator,; ProductLinearOperator,; SumLinearOperator,;
from probnum import backend
from probnum.linops._arithmetic_fallbacks import ScaledLinearOperator
from probnum.linops._linear_operator import Matrix
from probnum.problems.zoo.linalg import random_spd_matrix

import pytest


@pytest.fixture
def scalar():
    return 3.14


@pytest.fixture
def rand_spd_mat():
    rng_state = backend.random.rng_state(1237)
    return Matrix(random_spd_matrix(rng_state, shape=(4, 4)))


def test_scaled_linop(rand_spd_mat, scalar):
    with pytest.raises(TypeError):
        ScaledLinearOperator(np.random.rand(4, 4), scalar=scalar)
    with pytest.raises(TypeError):
        ScaledLinearOperator(rand_spd_mat, scalar=np.ones(4))

    scaled1 = ScaledLinearOperator(rand_spd_mat, scalar=0.0)
    scaled2 = ScaledLinearOperator(rand_spd_mat, scalar=scalar)

    with pytest.raises(np.linalg.LinAlgError):
        scaled1.inv()

    assert np.allclose(
        scaled2.inv().todense(), (1.0 / scalar) * scaled2._linop.inv().todense()
    )
