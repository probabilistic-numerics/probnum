"""Tests for functions generating random linear systems."""

import numpy as np
import pytest
import scipy.stats

from probnum import randvars
from probnum.problems.zoo.linalg import random_linear_system, random_spd_matrix


def test_custom_random_matrix(rng: np.random.Generator):
    random_unitary_matrix = lambda rng, dim: scipy.stats.unitary_group.rvs(
        dim=dim, random_state=rng
    )
    _ = random_linear_system(rng, random_unitary_matrix, dim=5)


def test_custom_solution_randvar(rng: np.random.Generator):
    n = 5
    x = randvars.Normal(mean=np.ones(n), cov=np.eye(n))
    _ = random_linear_system(rng=rng, matrix=random_spd_matrix, solution_rv=x, dim=n)


def test_incompatible_matrix_and_solution(rng: np.random.Generator):

    with pytest.raises(ValueError):
        _ = random_linear_system(
            rng=rng,
            matrix=random_spd_matrix,
            solution_rv=randvars.Normal(np.ones(2), np.eye(2)),
            dim=5,
        )
