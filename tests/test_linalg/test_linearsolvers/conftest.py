"""Test fixtures for probabilistic linear solvers."""

from typing import Iterator, Optional

import numpy as np
import pytest

import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import (
    LinearSolverState,
    ProbabilisticLinearSolver,
    beliefs,
    hyperparam_optim,
    observation_ops,
    policies,
    stop_criteria,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix


@pytest.fixture(
    params=[
        pytest.param(num_iters, id=f"iter{num_iters}") for num_iters in [1, 10, 100]
    ]
)
def num_iters(request) -> int:
    """Number of iterations of the linear solver.

    This is mostly used for test parameterization.
    """
    return request.param


###################
# (Prior Beliefs) #
###################


@pytest.fixture()
def prior(
    linsys_spd: LinearSystem, n: int, random_state: np.random.RandomState
) -> beliefs.LinearSystemBelief:
    """Prior belief over the linear system."""
    return beliefs.LinearSystemBelief.from_matrices(
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
) -> beliefs.WeakMeanCorrespondenceBelief:
    """Scalar weak mean correspondence belief."""
    return beliefs.WeakMeanCorrespondenceBelief.from_scalar(
        alpha=scalar, problem=linsys_spd
    )


@pytest.fixture()
def belief_groundtruth(linsys_spd: LinearSystem) -> beliefs.LinearSystemBelief:
    return beliefs.LinearSystemBelief(
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
    belief: beliefs.LinearSystemBelief,
    random_state: np.random.RandomState,
    solver_state: Optional[LinearSolverState] = None,
):
    action = rvs.Normal(
        0.0,
        1.0,
        random_state=random_state,
    ).sample((problem.A.shape[1], 1))
    action = action / np.linalg.norm(action)
    try:
        solver_state.actions.append(action)
    except AttributeError:
        pass
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


@pytest.fixture()
def actions(n: int, num_iters: int, random_state: np.random.RandomState) -> list:
    """Action chosen by a policy."""
    return [action[:, None] for action in (random_state.normal(size=(n, num_iters))).T]


#########################
# Observation Operators #
#########################


@pytest.fixture(
    params=[
        pytest.param(observation_op, id=observation_op_name)
        for (observation_op_name, observation_op) in zip(
            ["matvec"],
            [observation_ops.MatVecObservation()],
        )
    ]
)
def observation_op(request) -> observation_ops.ObservationOperator:
    """Observation operators of linear solvers."""
    return request.param


@pytest.fixture()
def matvec_observation(action: np.ndarray, linsys_spd: LinearSystem) -> np.ndarray:
    """Matrix-vector product observation for a given action."""
    return linsys_spd.A @ action


@pytest.fixture()
def matvec_observations(actions: np.ndarray, linsys_spd: LinearSystem) -> list:
    """Matrix-vector product observations for a given set of actions."""
    return list((linsys_spd.A @ np.array(actions).T).T)


###############################
# Hyperparameter Optimization #
###############################


@pytest.fixture(
    params=[
        pytest.param(calibration_method, id=calibration_method)
        for calibration_method in ["adhoc", "weightedmean", "gpkern"]
    ]
)
def calibration_method(request) -> str:
    return request.param


@pytest.fixture()
def uncertainty_calibration(
    calibration_method: str,
) -> hyperparam_optim.UncertaintyCalibration:
    return hyperparam_optim.UncertaintyCalibration(method=calibration_method)


##################
# Belief Updates #
##################


#####################
# Stopping Criteria #
#####################


def custom_stopping_criterion(
    problem: LinearSystem,
    belief: beliefs.LinearSystemBelief,
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
    linsys_spd: LinearSystem, prior: beliefs.LinearSystemBelief
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


@pytest.fixture()
def prob_linear_solver(
    prior: beliefs.LinearSystemBelief,
    policy: policies.Policy,
    observation_op: observation_ops.ObservationOperator,
    stopcrit: stop_criteria.StoppingCriterion,
):
    """Custom probabilistic linear solvers."""
    return ProbabilisticLinearSolver(
        prior=prior,
        policy=policy,
        observation_op=observation_op,
        stopping_criteria=[stop_criteria.MaxIterations(), stopcrit],
    )


@pytest.fixture()
def solve_iterator(
    prob_linear_solver: ProbabilisticLinearSolver,
    linsys_spd: LinearSystem,
    prior: beliefs.LinearSystemBelief,
    solver_state_init: LinearSolverState,
) -> Iterator:
    """Solver iterators of custom probabilistic linear solvers."""
    return prob_linear_solver.solve_iterator(
        problem=linsys_spd, belief=prior, solver_state=solver_state_init
    )


@pytest.fixture()
def conj_dir_method(
    prior: beliefs.LinearSystemBelief, stopcrit: stop_criteria.StoppingCriterion, n: int
):
    """Probabilistic linear solvers which are conjugate direction methods."""
    return ProbabilisticLinearSolver(
        prior=prior,
        policy=policies.ConjugateDirections(),
        observation_op=observation_ops.MatVecObservation(),
        stopping_criteria=[
            stop_criteria.MaxIterations(maxiter=n),
            stop_criteria.Residual(),
            stopcrit,
        ],
    )


@pytest.fixture(
    params=[pytest.param(alpha, id=f"alpha{alpha}") for alpha in [0.01, 1.0, 3.5]]
)
def conj_grad_method(
    request,
    uncertainty_calibration: hyperparam_optim.UncertaintyCalibration,
    linsys_spd: LinearSystem,
):
    """Probabilistic linear solvers which are conjugate gradient methods."""
    return ProbabilisticLinearSolver(
        prior=beliefs.WeakMeanCorrespondenceBelief.from_scalar(
            alpha=request.param,
            problem=linsys_spd,
            calibration_method=uncertainty_calibration,
        ),
        policy=policies.ConjugateDirections(),
        observation_op=observation_ops.MatVecObservation(),
        optimize_hyperparams=True,
        stopping_criteria=[stop_criteria.MaxIterations(), stop_criteria.Residual()],
    )
