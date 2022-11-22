"""Test cases defining linear systems to be solved."""

from probnum import backend, problems
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix

from pytest_cases import case


@case(tags=["sym", "posdef"])
def case_random_spd_linsys(
    ncols: int,
) -> problems.LinearSystem:
    rng_state = backend.random.rng_state(1)
    rng_state_A, rng_state_x = backend.random.split(rng_state)
    A = random_spd_matrix(rng_state=rng_state_A, shape=(ncols, ncols))
    x = backend.random.standard_normal(rng_state=rng_state_x, shape=(ncols,))
    b = A @ x
    return problems.LinearSystem(A=A, b=b, solution=x)


@case(tags=["sym", "posdef", "sparse"])
def case_random_sparse_spd_linsys(
    ncols: int,
) -> problems.LinearSystem:
    rng_state = backend.random.rng_state(1)
    rng_state_A, rng_state_x = backend.random.split(rng_state)
    A = random_sparse_spd_matrix(
        rng_state=rng_state_A, shape=(ncols, ncols), density=0.1
    )
    x = backend.random.standard_normal(rng_state=rng_state_x, shape=(ncols,))
    b = A @ x
    return problems.LinearSystem(A=A, b=b, solution=x)
