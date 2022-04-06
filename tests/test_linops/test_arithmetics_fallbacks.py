"""Tests for linear operator arithmetics fallbacks."""

import numpy as np
import pytest

from probnum.linops._arithmetic_fallbacks import (
    # NegatedLinearOperator,; ProductLinearOperator,; SumLinearOperator,
    ScaledLinearOperator,
)
from probnum.linops._linear_operator import Matrix
from probnum.problems.zoo.linalg import random_spd_matrix


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def scalar():
    return 3.14


@pytest.fixture
def rand_spd_mat(rng):
    return Matrix(random_spd_matrix(rng, dim=4))


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
