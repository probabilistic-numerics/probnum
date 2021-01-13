"""Test fixtures for probabilistic linear solvers."""

from functools import partial

import numpy as np
import pytest

import probnum
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import policies

###################
# (Prior Beliefs) #
###################


############
# Policies #
############


def custom_policy(problem, belief, random_state, solver_state=None):
    action = rvs.Normal(
        np.zeros((problem.A.shape[1], 1)),
        np.eye(problem.A.shape[1]),
        random_state=random_state,
    ).sample()
    return action, solver_state


@pytest.fixture(
    params=[
        pytest.param(policy, id=policy_name)
        for (policy_name, policy) in zip(
            ["conjugatedirs" "thompson", "exploreexploit", "custom"],
            [
                policies.ConjugateDirections,
                policies.ThompsonSampling(random_state=1),
                policies.ExploreExploit(random_state=1),
                policies.Policy(
                    policy=custom_policy, is_deterministic=False, random_state=1
                ),
            ],
        )
    ]
)
def policy(request, random_state: np.random.RandomState) -> policies.Policy:
    """Stochastic policy returning an action."""
    return request.param


#########################
# Observation Operators #
#########################


##################
# Belief Updates #
##################


################################
# Probabilistic Linear Solvers #
################################
