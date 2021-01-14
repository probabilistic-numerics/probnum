"""Test fixtures for probabilistic linear solvers."""

from typing import Optional

import numpy as np
import pytest

import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import (
    LinearSolverState,
    observation_ops,
    policies,
    stop_criteria,
)
from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix

###################
# (Prior Beliefs) #
###################


@pytest.fixture()
def prior(
    linsys_spd: LinearSystem, n: int, random_state: np.random.RandomState
) -> LinearSystemBelief:
    """Prior belief over the linear system."""
    return LinearSystemBelief.from_matrices(
        A0=random_spd_matrix(dim=n, random_state=random_state),
        Ainv0=random_spd_matrix(dim=n, random_state=random_state),
        problem=linsys_spd,
    )


@pytest.fixture(
    params=[pytest.param(scalar, id=f"alpha{scalar}") for scalar in [0.1, 1.0, 10]]
)
def scalar_weakmeancorr_prior(
    scalar: float,
    linsys_spd: LinearSystem,
) -> WeakMeanCorrespondenceBelief:
    """Scalar weak mean correspondence belief."""
    return WeakMeanCorrespondenceBelief.from_scalar(alpha=scalar, problem=linsys_spd)


@pytest.fixture()
def belief_groundtruth(linsys_spd: LinearSystem) -> LinearSystemBelief:
    return LinearSystemBelief(
        x=rvs.Constant(linsys_spd.solution),
        A=rvs.Constant(linsys_spd.A),
        Ainv=rvs.Constant(np.linalg.inv(linsys_spd.A)),
        b=rvs.Constant(linsys_spd.b),
    )


############
# Policies #
############


def custom_policy(
    problem: LinearSystem,
    belief: LinearSystemBelief,
    random_state: np.random.RandomState,
    solver_state: Optional[LinearSolverState] = None,
):
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
            ["conjugatedirs", "thompson", "exploreexploit", "custom"],
            [
                policies.ConjugateDirections(),
                policies.ThompsonSampling(random_state=1),
                policies.ExploreExploit(random_state=1),
                policies.Policy(
                    policy=custom_policy, is_deterministic=False, random_state=1
                ),
            ],
        )
    ]
)
def policy(request) -> policies.Policy:
    """Policies of linear solvers returning an action."""
    return request.param


@pytest.fixture()
def action(n: int, random_state: np.random.RandomState) -> np.ndarray:
    """Action chosen by a policy."""
    return random_state.normal(size=(n, 1))


#########################
# Observation Operators #
#########################


@pytest.fixture(
    params=[
        pytest.param(observation_op, id=observation_op_name)
        for (observation_op_name, observation_op) in zip(
            ["matvec"],
            [observation_ops.MatrixMultObservation()],
        )
    ]
)
def observation_op(request) -> observation_ops.ObservationOperator:
    """Observation operators of linear solvers."""
    return request.param


@pytest.fixture()
def matmul_obs(action: np.ndarray, linsys_spd: LinearSystem) -> np.ndarray:
    """Matrix-vector product observation for a given action."""
    return linsys_spd @ action


##################
# Belief Updates #
##################


#####################
# Stopping Criteria #
#####################


def custom_stopping_criterion(
    problem: LinearSystem,
    belief: LinearSystemBelief,
    solver_state: Optional[LinearSolverState] = None,
):
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
                stop_criteria.Residual(),
                stop_criteria.PosteriorContraction(),
                stop_criteria.StoppingCriterion(
                    stopping_criterion=custom_stopping_criterion
                ),
            ],
        )
    ]
)
def stopcrit(request) -> stop_criteria.StoppingCriterion:
    """Observation operators of linear solvers."""
    return request.param


################################
# Probabilistic Linear Solvers #
################################


@pytest.fixture()
def solver_state_init(
    linsys_spd: LinearSystem, prior: LinearSystemBelief
) -> LinearSolverState:
    return LinearSolverState(
        actions=[],
        observations=[],
        iteration=0,
        residual=linsys_spd.A @ prior.x.mean - linsys_spd.b,
        action_obs_innerprods=[],
        log_rayleigh_quotients=[],
        step_sizes=[],
        has_converged=False,
        stopping_criterion=None,
    )
