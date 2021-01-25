"""Fixtures for polices of probabilistic linear solvers."""
from typing import Optional

import numpy as np
import pytest

import probnum.random_variables as rvs
from probnum.linalg.solvers import LinearSolverState, beliefs, policies
from probnum.linalg.solvers.data import LinearSolverAction
from probnum.problems import LinearSystem


def custom_policy(
    problem: LinearSystem,
    belief: beliefs.LinearSystemBelief,
    random_state: np.random.RandomState,
    solver_state: Optional[LinearSolverState] = None,
):
    """Custom stochastic linear solver policy."""
    action = rvs.Normal(
        0.0,
        1.0,
        random_state=random_state,
    ).sample((problem.A.shape[1], 1))
    action = action / np.linalg.norm(action)

    return LinearSolverAction(A=action)


@pytest.fixture(
    params=[
        pytest.param(policy, id=policy_name)
        for (policy_name, policy) in zip(
            ["conjugatedirs", "thompson", "exploreexploit", "maxnormcol", "custom"],
            [
                policies.ConjugateDirections(),
                policies.ThompsonSampling(random_state=1),
                policies.ExploreExploit(random_state=1),
                policies.MaxSupNormColumn(),
                policies.Policy(
                    policy=custom_policy, is_deterministic=False, random_state=1
                ),
            ],
        )
    ],
    name="policy",
)
def fixture_policy(request) -> policies.Policy:
    """Policies of linear solvers returning an action."""
    return request.param


@pytest.fixture
def maxsupnormcol_policy() -> policies.MaxSupNormColumn:
    """Maximum column supremum-norm policy."""
    return policies.MaxSupNormColumn()
