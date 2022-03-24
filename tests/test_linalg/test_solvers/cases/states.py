"""Probabilistic linear solver state test cases."""

import numpy as np
from pytest_cases import case

from probnum import backend, linalg, linops, randvars
from probnum.problems.zoo.linalg import random_linear_system, random_spd_matrix

# Problem
n = 10
linsys = random_linear_system(42, matrix=random_spd_matrix, dim=n)

# Prior
Ainv = randvars.Normal(
    mean=linops.Identity(n), cov=linops.SymmetricKronecker(linops.Identity(n))
)
b = randvars.Constant(backend.to_numpy(linsys.b))
prior = linalg.solvers.beliefs.LinearSystemBelief(
    A=randvars.Constant(linsys.A),
    Ainv=Ainv,
    x=Ainv @ b,
    b=b,
)


@case(tags=["initial"])
def case_initial_state():
    """Initial state of a linear solver."""
    return linalg.solvers.LinearSolverState(problem=linsys, prior=prior)


@case(tags=["has_action"])
def case_state(
    rng: np.random.Generator,
):
    """State of a linear solver."""
    state = linalg.solvers.LinearSolverState(problem=linsys, prior=prior)
    state.action = rng.standard_normal(size=state.problem.A.shape[1])

    return state


@case(tags=["has_action", "has_observation", "matrix_based"])
def case_state_matrix_based(
    rng: np.random.Generator,
):
    """State of a matrix-based linear solver."""
    prior = linalg.solvers.beliefs.LinearSystemBelief(
        A=randvars.Normal(
            mean=linops.Matrix(linsys.A),
            cov=linops.Kronecker(A=linops.Identity(n), B=linops.Identity(n)),
        ),
        x=(Ainv @ b[:, None]).reshape((n,)),
        Ainv=randvars.Normal(
            mean=linops.Identity(n),
            cov=linops.Kronecker(A=linops.Identity(n), B=linops.Identity(n)),
        ),
        b=b,
    )
    state = linalg.solvers.LinearSolverState(problem=linsys, prior=prior)
    state.action = rng.standard_normal(size=state.problem.A.shape[1])
    state.observation = rng.standard_normal(size=state.problem.A.shape[1])

    return state


@case(tags=["has_action", "has_observation", "symmetric_matrix_based"])
def case_state_symmetric_matrix_based(
    rng: np.random.Generator,
):
    """State of a symmetric matrix-based linear solver."""
    prior = linalg.solvers.beliefs.LinearSystemBelief(
        A=randvars.Normal(
            mean=linops.Matrix(linsys.A),
            cov=linops.SymmetricKronecker(A=linops.Identity(n)),
        ),
        x=(Ainv @ b[:, None]).reshape((n,)),
        Ainv=randvars.Normal(
            mean=linops.Identity(n),
            cov=linops.SymmetricKronecker(A=linops.Identity(n)),
        ),
        b=b,
    )
    state = linalg.solvers.LinearSolverState(problem=linsys, prior=prior)
    state.action = rng.standard_normal(size=state.problem.A.shape[1])
    state.observation = rng.standard_normal(size=state.problem.A.shape[1])

    return state


@case(tags=["has_action", "has_observation", "solution_based"])
def case_state_solution_based(
    rng: np.random.Generator,
):
    """State of a solution-based linear solver."""
    initial_state = linalg.solvers.LinearSolverState(problem=linsys, prior=prior)
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
    state = linalg.solvers.LinearSolverState(problem=linsys, prior=belief)
    return state
