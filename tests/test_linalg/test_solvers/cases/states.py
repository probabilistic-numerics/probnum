"""Probabilistic linear solver state test cases."""

import numpy as np

from probnum import linalg, linops, problems, randvars
from probnum.problems.zoo.linalg import random_spd_matrix


def case_linear_solver_state(
    rng: np.random.Generator,
):
    """State of a linear solver."""
    n = 10
    linsys = problems.LinearSystem.from_matrix(
        random_spd_matrix(dim=n, random_state=rng), rng=rng
    )
    prior = linalg.solvers.beliefs.LinearSystemBelief(
        A=randvars.Constant(linsys.A),
        Ainv=None,
        x=randvars.Normal(
            mean=np.zeros(linsys.A.shape[1]), cov=linops.Identity(shape=linsys.A.shape)
        ),
        b=randvars.Constant(linsys.b),
    )
    return linalg.solvers.LinearSolverState(problem=linsys, prior=prior, rng=rng)
