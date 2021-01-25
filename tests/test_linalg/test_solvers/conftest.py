"""Test fixtures for probabilistic linear solvers."""

from typing import Iterator

import numpy as np
import pytest

from probnum.linalg.solvers import (
    LinearSolverCache,
    LinearSolverInfo,
    LinearSolverState,
    ProbabilisticLinearSolver,
    beliefs,
    observation_ops,
    policies,
    stop_criteria,
)
from probnum.linalg.solvers.data import (
    LinearSolverAction,
    LinearSolverData,
    LinearSolverObservation,
)
from probnum.problems import LinearSystem


@pytest.fixture(
    params=[
        pytest.param(num_iters, id=f"iter{num_iters}") for num_iters in [1, 10, 100]
    ],
    name="num_iters",
)
def fixture_num_iters(request) -> int:
    """Number of iterations of a linear solver."""
    return request.param


############
# Data #
############


@pytest.fixture(name="action")
def fixture_action(n: int, random_state: np.random.RandomState) -> LinearSolverAction:
    """Action chosen by a policy."""
    return LinearSolverAction(A=random_state.normal(size=(n, 1)))


@pytest.fixture()
def matvec_observation(
    action: LinearSolverAction, linsys_spd: LinearSystem
) -> LinearSolverObservation:
    """Matrix-vector product observation for a given action."""
    return LinearSolverObservation(A=linsys_spd.A @ action.A, b=linsys_spd.b)


@pytest.fixture
def solver_data(
    n: int,
    num_iters: int,
    linsys_spd: LinearSystem,
    random_state: np.random.RandomState,
):
    """Data collected by a linear solver."""
    actions = [
        LinearSolverAction(A=s[:, None])
        for s in (random_state.normal(size=(n, num_iters))).T
    ]
    matvec_observations = [
        LinearSolverObservation(A=linsys_spd.A @ action.A, b=linsys_spd.b)
        for action in actions
    ]
    return LinearSolverData(actions=actions, observations=matvec_observations)


################
# Solver State #
################


@pytest.fixture(name="solver_info")
def fixture_solver_info(num_iters: int, stopcrit: stop_criteria.StoppingCriterion):
    """Convergence information of a linear solver."""
    return LinearSolverInfo(
        iteration=num_iters,
        has_converged=True,
        stopping_criterion=stopcrit,
    )


@pytest.fixture
def solver_cache(linsys: LinearSystem, belief: beliefs.LinearSystemBelief):
    """Miscellaneous quantities computed (and cached) by a linear solver."""
    return LinearSolverCache(problem=linsys, belief=belief)


@pytest.fixture(name="solver_state_init")
def fixture_solver_state_init(
    linsys_spd: LinearSystem, prior: beliefs.LinearSystemBelief
) -> LinearSolverState:
    """Initial solver state of a probabilistic linear solver."""
    return LinearSolverState(
        problem=linsys_spd,
        belief=prior,
    )


################################
# Probabilistic Linear Solvers #
################################


@pytest.fixture(name="prob_linear_solver")
def fixture_prob_linear_solver(
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
    # uncertainty_calibration: hyperparam_optim.UncertaintyCalibration,
    linsys_spd: LinearSystem,
):
    """Probabilistic linear solvers which are conjugate gradient methods."""
    return ProbabilisticLinearSolver(
        prior=beliefs.WeakMeanCorrespondenceBelief.from_scalar(
            scalar=request.param,
            problem=linsys_spd,
            # calibration_method=uncertainty_calibration,
        ),
        policy=policies.ConjugateDirections(),
        observation_op=observation_ops.MatVecObservation(),
        stopping_criteria=[stop_criteria.MaxIterations(), stop_criteria.Residual()],
    )
