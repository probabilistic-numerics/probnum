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
prior = linalg.solvers.beliefs.LinearSystemBelief(
    A=randvars.Constant(linsys.A),
    Ainv=None,
    x=randvars.Normal(
        mean=np.zeros(linsys.A.shape[1]), cov=linops.Identity(shape=linsys.A.shape)
    ),
    b=randvars.Constant(linsys.b),
)


@case(tags=["initial"])
def case_initial_state(
    rng: np.random.Generator,
):
    """Initial state of a linear solver."""
    return linalg.solvers.ProbabilisticLinearSolverState(
        problem=linsys, prior=prior, rng=rng
    )


@case(tags=["has_action"])
def case_state(
    rng: np.random.Generator,
):
    """State of a linear solver."""
    initial_state = linalg.solvers.ProbabilisticLinearSolverState(
        problem=linsys, prior=prior, rng=rng
    )
    initial_state.action = rng.standard_normal(size=initial_state.problem.A.shape[1])

    return initial_state
