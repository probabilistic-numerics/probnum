"""Tests for functions generating random linear systems."""

from probnum import backend, randvars
from probnum.problems.zoo.linalg import random_linear_system, random_spd_matrix

import pytest


def test_custom_random_matrix():
    rng_state = backend.random.rng_state(305985)
    random_unitary_matrix = lambda rng_state, n: backend.random.uniform_so_group(
        n=n, rng_state=rng_state
    )
    _ = random_linear_system(rng_state, random_unitary_matrix, n=5)


def test_custom_solution_randvar():
    n = 5
    rng_state = backend.random.rng_state(3453)
    x = randvars.Normal(mean=backend.ones(n), cov=backend.eye(n))
    _ = random_linear_system(
        rng_state=rng_state, matrix=random_spd_matrix, solution_rv=x, shape=(n, n)
    )


def test_incompatible_matrix_and_solution():
    rng_state = backend.random.rng_state(3453)

    with pytest.raises(ValueError):
        _ = random_linear_system(
            rng_state=rng_state,
            matrix=random_spd_matrix,
            solution_rv=randvars.Normal(backend.ones(2), backend.eye(2)),
            shape=(5, 5),
        )
