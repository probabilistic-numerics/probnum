"""Test fixtures for probabilistic linear solvers."""

from typing import Iterator

import numpy as np
import pytest

import probnum.linops as linops
import probnum.random_variables as rvs
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
from probnum.problems import LinearSystem, NoisyLinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix


@pytest.fixture(
    params=[
        pytest.param(num_iters, id=f"iter{num_iters}") for num_iters in [1, 10, 100]
    ],
    name="num_iters",
)
def fixture_num_iters(request) -> int:
    """Number of iterations of a linear solver."""
    return request.param


###########
# Beliefs #
###########


@pytest.fixture(name="prior")
def fixture_prior(
    linsys_spd: LinearSystem, n: int, random_state: np.random.RandomState
) -> beliefs.SymmetricNormalLinearSystemBelief:
    """Symmetric normal prior belief about the linear system."""
    return beliefs.SymmetricNormalLinearSystemBelief.from_matrices(
        A0=random_spd_matrix(dim=n, random_state=random_state),
        Ainv0=random_spd_matrix(dim=n, random_state=random_state),
        problem=linsys_spd,
    )


@pytest.fixture(
    params=[
        pytest.param(bc, id=bc.__name__)
        for bc in [
            beliefs.LinearSystemBelief,
            beliefs.SymmetricNormalLinearSystemBelief,
            beliefs.WeakMeanCorrespondenceBelief,
            beliefs.NoisySymmetricNormalLinearSystemBelief,
        ]
    ],
    name="belief_class",
)
def fixture_belief_class(request):
    """A linear system belief class."""
    return request.param


@pytest.fixture(name="belief")
def fixture_belief(belief_class, mat, linsys):
    """Linear system beliefs."""
    return belief_class.from_inverse(Ainv0=linops.aslinop(mat), problem=linsys)


@pytest.fixture()
def belief_groundtruth(linsys_spd: LinearSystem) -> beliefs.LinearSystemBelief:
    """Belief equalling the true solution of the linear system."""
    return beliefs.LinearSystemBelief(
        x=rvs.Constant(linsys_spd.solution),
        A=rvs.Constant(linsys_spd.A),
        Ainv=rvs.Constant(np.linalg.inv(linsys_spd.A)),
        b=rvs.Constant(linsys_spd.b),
    )


@pytest.fixture(
    params=[
        pytest.param(inv, id=inv[0])
        for inv in [
            (
                "weakmeancorr_scalar",
                beliefs.WeakMeanCorrespondenceBelief,
                lambda n: linops.ScalarMult(scalar=1.0, shape=(n, n)),
            ),
            (
                "symmnormal_dense",
                beliefs.SymmetricNormalLinearSystemBelief,
                lambda n: rvs.Normal(
                    mean=random_spd_matrix(n, random_state=42),
                    cov=linops.SymmetricKronecker(
                        A=random_spd_matrix(n, random_state=1)
                    ),
                ),
            ),
            (
                "symmnormal_sparse",
                beliefs.SymmetricNormalLinearSystemBelief,
                lambda n: rvs.Normal(
                    mean=random_sparse_spd_matrix(n, density=0.01, random_state=42),
                    cov=linops.SymmetricKronecker(
                        A=random_sparse_spd_matrix(n, density=0.01, random_state=1)
                    ),
                ),
            ),
        ]
    ],
    name="symm_belief",
)
def fixture_symm_belief(
    request, n: int, linsys_spd: LinearSystem
) -> beliefs.SymmetricNormalLinearSystemBelief:
    """Symmetric normal linear system belief."""
    return request.param[1].from_inverse(Ainv0=request.param[2](n), problem=linsys_spd)


##################
# Linear Systems #
##################


@pytest.fixture(name="linsys_matnoise")
def fixture_linsys_matnoise(
    eps: float,
    linsys_spd: LinearSystem,
    random_state: np.random.RandomState,
    prior: beliefs.LinearSystemBelief,
) -> NoisyLinearSystem:
    r"""Linear system with noise-corrupted matrix :math:`A + E` such that :math:`E
    \sim \mathcal{N}(0, \varepsilon^2 W_0 \otimes_s W_0)`"""
    return NoisyLinearSystem.from_randvars(
        A=rvs.Normal(
            mean=linsys_spd.A,
            cov=linops.SymmetricKronecker(eps * prior.A.cov.A),
            random_state=random_state,
        ),
        b=rvs.asrandvar(linsys_spd.b),
        solution=linsys_spd.solution,
    )


################
# Problem Data #
################


@pytest.fixture(name="action")
def fixture_action(n: int, random_state: np.random.RandomState) -> LinearSolverAction:
    """Action chosen by a policy."""
    return LinearSolverAction(actA=random_state.normal(size=(n, 1)))


@pytest.fixture()
def matvec_observation(
    action: LinearSolverAction, linsys_spd: LinearSystem
) -> LinearSolverObservation:
    """Matrix-vector product observation for a given action."""
    return LinearSolverObservation(obsA=linsys_spd.A @ action.actA, obsb=linsys_spd.b)


@pytest.fixture
def solver_data(
    n: int,
    num_iters: int,
    linsys_spd: LinearSystem,
    random_state: np.random.RandomState,
):
    """Data collected by a linear solver."""
    actions = [
        LinearSolverAction(actA=s[:, None])
        for s in (random_state.normal(size=(n, num_iters))).T
    ]
    matvec_observations = [
        LinearSolverObservation(obsA=linsys_spd.A @ action.actA, obsb=linsys_spd.b)
        for action in actions
    ]
    return LinearSolverData(actions=actions, observations=matvec_observations)


@pytest.fixture
def noisy_solver_data(
    n: int,
    num_iters: int,
    linsys_matnoise: NoisyLinearSystem,
    random_state: np.random.RandomState,
):
    """Data collected by a linear solver on a noisy problem."""
    actions = [
        LinearSolverAction(actA=s[:, None])
        for s in (random_state.normal(size=(n, num_iters))).T
    ]
    sampled_systems = linsys_matnoise.sample(size=len(actions))
    matvec_observations = [
        LinearSolverObservation(
            obsA=action_system_pair[1][0] @ action_system_pair[0].actA,
            obsb=action_system_pair[1][1],
        )
        for action_system_pair in zip(actions, sampled_systems)
    ]
    return LinearSolverData(actions=actions, observations=matvec_observations)


################
# Solver State #
################


@pytest.fixture(
    params=[
        pytest.param(stopcrit, id=stopcrit_name)
        for (stopcrit_name, stopcrit) in zip(
            ["maxiter", "residual", "uncertainty"],
            [
                stop_criteria.MaxIterations(),
                stop_criteria.Residual(),
                stop_criteria.PosteriorContraction(),
            ],
        )
    ],
    name="solver_info",
)
def fixture_solver_info(
    request,
    num_iters: int,
):
    """Convergence information of a linear solver."""
    return LinearSolverInfo(
        iteration=num_iters,
        has_converged=True,
        stopping_criterion=request.param,
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
        observation_op=observation_ops.MatVec(),
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
        observation_op=observation_ops.MatVec(),
        stopping_criteria=[stop_criteria.MaxIterations(), stop_criteria.Residual()],
    )
