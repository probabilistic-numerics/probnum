"""Probabilistic linear solver state test cases."""

import numpy as np
from pytest_cases import parametrize_with_cases

from probnum import linalg, linops, randvars
from probnum.problems.zoo.linalg import random_linear_system, random_spd_matrix


def case_initial_state(
    rng: np.random.Generator,
):
    """Initial state of a linear solver."""
    # Problem
    n = 5
    linsys = random_linear_system(rng=rng, matrix=random_spd_matrix, dim=n)

    # Prior
    prior = linalg.solvers.beliefs.LinearSystemBelief(
        A=randvars.Constant(linsys.A),
        Ainv=None,
        x=randvars.Normal(
            mean=np.zeros(linsys.A.shape[1]), cov=linops.Identity(shape=linsys.A.shape)
        ),
        b=randvars.Constant(linsys.b),
    )

    # State
    solver_state = linalg.solvers.ProbabilisticLinearSolverState(
        problem=linsys, prior=prior, rng=rng
    )

    return solver_state


@parametrize_with_cases("initial_state", cases=case_initial_state)
def case_state(initial_state: linalg.solvers.ProbabilisticLinearSolverState):
    """State of a linear solver."""
    initial_state.action = initial_state.rng.standard_normal(
        size=initial_state.problem.A.shape[1]
    )

    return initial_state
