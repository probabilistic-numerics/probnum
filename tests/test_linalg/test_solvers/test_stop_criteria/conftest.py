"""Test fixtures for stopping criteria of probabilistic linear solvers."""

from typing import Optional

import numpy as np
import pytest

from probnum.linalg.solvers import LinearSolverState, beliefs, stop_criteria
from probnum.problems import LinearSystem


def custom_stopping_criterion(
    problem: LinearSystem,
    belief: beliefs.LinearSystemBelief,
    solver_state: Optional[LinearSolverState] = None,
):
    """Custom stopping criterion of a probabilistic linear solver."""
    _has_converged = (
        np.ones((1, belief.A.shape[0]))
        @ (belief.Ainv @ np.ones((belief.A.shape[0], 1)))
    ).cov.item() < 10 ** -3
    try:
        return solver_state.iteration >= 100 or _has_converged
    except AttributeError:
        return _has_converged


@pytest.fixture(
    params=[
        pytest.param(stopcrit, id=stopcrit_name)
        for (stopcrit_name, stopcrit) in zip(
            ["maxiter", "residual", "uncertainty", "custom"],
            [
                stop_criteria.MaxIterations(),
                stop_criteria.ResidualNorm(),
                stop_criteria.PosteriorContraction(),
                stop_criteria.StoppingCriterion(
                    stopping_criterion=custom_stopping_criterion
                ),
            ],
        )
    ],
    name="stopcrit",
)
def fixture_stopcrit(request) -> stop_criteria.StoppingCriterion:
    """Stopping criteria of linear solvers."""
    return request.param
