"""Probabilistic linear solver state test cases."""

import numpy as np
from pytest_cases import case

from probnum import linalg, linops, randvars
from probnum.problems.zoo.linalg import random_linear_system, random_spd_matrix

# Problem
n = 10
linsys = random_linear_system(
    rng=np.random.default_rng(42), matrix=random_spd_matrix, dim=n
)

# Prior
Ainv = randvars.Normal(
    mean=linops.Identity(n), cov=linops.SymmetricKronecker(linops.Identity(n))
)
b = randvars.Constant(linsys.b)
prior = linalg.solvers.beliefs.LinearSystemBelief(
    A=randvars.Normal(linsys.A, cov=linops.SymmetricKronecker(linops.Identity(n))),
    Ainv=Ainv,
    x=(Ainv @ b[:, None]).reshape(
        (n,)
    ),  # TODO: This can be replaced by Ainv @ b once https://github.com/probabilistic-numerics/probnum/issues/456 is fixed
    b=b,
)


@case(tags=["initial"])
def case_initial_state(
    rng: np.random.Generator,
):
    """Initial state of a linear solver."""
    return linalg.solvers.LinearSolverState(problem=linsys, prior=prior, rng=rng)


@case(tags=["has_action"])
def case_state(
    rng: np.random.Generator,
):
    """State of a linear solver."""
    initial_state = linalg.solvers.LinearSolverState(
        problem=linsys, prior=prior, rng=rng
    )
    initial_state.action = rng.standard_normal(size=initial_state.problem.A.shape[1])

    return initial_state


@case(tags=["has_action", "has_observation", "matrix_based"])
def case_state_matrix_based(
    rng: np.random.Generator,
):
    """State of a solution-based linear solver."""
    initial_state = linalg.solvers.LinearSolverState(
        problem=linsys, prior=prior, rng=rng
    )
    initial_state.action = rng.standard_normal(size=initial_state.problem.A.shape[1])
    initial_state.observation = rng.standard_normal(
        size=initial_state.problem.A.shape[1]
    )

    return initial_state


@case(tags=["has_action", "has_observation", "solution_based"])
def case_state_solution_based(
    rng: np.random.Generator,
):
    """State of a solution-based linear solver."""
    initial_state = linalg.solvers.LinearSolverState(
        problem=linsys, prior=prior, rng=rng
    )
    initial_state.action = rng.standard_normal(size=initial_state.problem.A.shape[1])
    initial_state.observation = rng.standard_normal()

    return initial_state


def case_state_converged(
    rng: np.random.Generator,
):
    """State of a linear solver, which has converged at initialization."""
    belief = linalg.solvers.beliefs.LinearSystemBelief(
        A=randvars.Constant(linsys.A),
        Ainv=randvars.Constant(linops.aslinop(linsys.A).inv().todense()),
        x=randvars.Constant(linsys.solution),
        b=randvars.Constant(linsys.b),
    )
    state = linalg.solvers.LinearSolverState(problem=linsys, prior=belief, rng=rng)
    return state
