"""Test fixtures for probabilistic linear solvers."""

from typing import Iterator, Optional

import numpy as np
import pytest

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers import (
    LinearSolverState,
    ProbabilisticLinearSolver,
    belief_updates,
    beliefs,
    hyperparam_optim,
    observation_ops,
    policies,
    stop_criteria,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix


@pytest.fixture(
    params=[
        pytest.param(num_iters, id=f"iter{num_iters}") for num_iters in [1, 10, 100]
    ],
    name="num_iters",
)
def fixture_num_iters(request) -> int:
    """Number of iterations of the linear solver.

    This is mostly used for test parameterization.
    """
    return request.param


###################
# (Prior) Beliefs #
###################


@pytest.fixture(
    params=[
        pytest.param(bc, id=bc.__name__)
        for bc in [
            beliefs.LinearSystemBelief,
            beliefs.SymmetricLinearSystemBelief,
            beliefs.WeakMeanCorrespondenceBelief,
            beliefs.NoisyLinearSystemBelief,
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
    return belief_class.from_inverse(Ainv0=mat, problem=linsys)


@pytest.fixture(name="prior")
def fixture_prior(
    linsys_spd: LinearSystem, n: int, random_state: np.random.RandomState
) -> beliefs.SymmetricLinearSystemBelief:
    """Symmetric normal prior belief over the linear system."""
    return beliefs.SymmetricLinearSystemBelief.from_matrices(
        A0=random_spd_matrix(dim=n, random_state=random_state),
        Ainv0=random_spd_matrix(dim=n, random_state=random_state),
        problem=linsys_spd,
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
                beliefs.SymmetricLinearSystemBelief,
                lambda n: rvs.Normal(
                    mean=random_spd_matrix(n, random_state=42),
                    cov=linops.SymmetricKronecker(
                        A=random_spd_matrix(n, random_state=1)
                    ),
                ),
            ),
            (
                "symmnormal_sparse",
                beliefs.SymmetricLinearSystemBelief,
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
) -> beliefs.SymmetricLinearSystemBelief:
    """Symmetric normal linear system belief."""
    return request.param[1].from_inverse(Ainv0=request.param[2](n), problem=linsys_spd)


@pytest.fixture(
    params=[
        pytest.param(inv, id=inv[0])
        for inv in [
            ("scalar", lambda n: linops.ScalarMult(scalar=1.0, shape=(n, n))),
            (
                "spd",
                lambda n: linops.MatrixMult(A=random_spd_matrix(n, random_state=42)),
            ),
            (
                "sparse",
                lambda n: linops.MatrixMult(
                    A=random_sparse_spd_matrix(n, density=0.1, random_state=42)
                ),
            ),
        ]
    ],
    name="weakmeancorr_belief",
)
def fixture_weakmeancorr_belief(
    request, n: int, linsys_spd: LinearSystem, actions: list, matvec_observations: list
):
    """Symmetric Gaussian weak mean correspondence belief."""
    return beliefs.WeakMeanCorrespondenceBelief.from_inverse(
        Ainv0=request.param[1](n),
        actions=actions,
        observations=matvec_observations,
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
        scalar=scalar, problem=linsys_spd
    )


@pytest.fixture()
def belief_groundtruth(linsys_spd: LinearSystem) -> beliefs.LinearSystemBelief:
    """Belief equalling the true solution of the linear system."""
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
    """Custom stochastic linear solver policy."""
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
    ],
    name="policy",
)
def fixture_policy(request) -> policies.Policy:
    """Policies of linear solvers returning an action."""
    return request.param


@pytest.fixture(name="action")
def fixture_action(n: int, random_state: np.random.RandomState) -> np.ndarray:
    """Action chosen by a policy."""
    return random_state.normal(size=(n, 1))


@pytest.fixture(name="actions")
def fixture_actions(
    n: int, num_iters: int, random_state: np.random.RandomState
) -> list:
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
    ],
    name="observation_op",
)
def fixture_observation_op(request) -> observation_ops.ObservationOperator:
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
    ],
    name="calibration_method",
)
def fixture_calibration_method(request) -> str:
    """Names of available uncertainty calibration methods."""
    return request.param


@pytest.fixture(name="uncertainty_calibration")
def fixture_uncertainty_calibration(
    calibration_method: str,
) -> hyperparam_optim.UncertaintyCalibration:
    """Uncertainty calibration method for probabilistic linear solvers."""
    return hyperparam_optim.UncertaintyCalibration(method=calibration_method)


##################
# Belief Updates #
##################


@pytest.fixture(
    params=[
        pytest.param(bel_upd, id=bel_upd[0])
        for bel_upd in [
            (
                "symmlin",
                beliefs.SymmetricLinearSystemBelief,
                belief_updates.SymmetricNormalLinearObsBeliefUpdate,
            ),
            (
                "weakmeancorrlin",
                beliefs.WeakMeanCorrespondenceBelief,
                belief_updates.WeakMeanCorrLinearObsBeliefUpdate,
            ),
        ]
    ],
    name="linobs_belief_update",
)
def fixture_linobs_belief_update(
    request,
    n: int,
    random_state: np.random.RandomState,
    linsys_spd: LinearSystem,
    action: np.ndarray,
    matvec_observation: np.ndarray,
) -> belief_updates.BeliefUpdate:
    belief = request.param[1].from_inverse(
        linops.MatrixMult(random_spd_matrix(dim=n, random_state=random_state)),
        problem=linsys_spd,
    )
    """Belief update for linear observations."""
    return request.param[2](
        problem=linsys_spd,
        belief=belief,
        actions=action,
        observations=matvec_observation,
    )


@pytest.fixture(
    params=[
        pytest.param(noise_cov, id=f"noise{noise_cov}")
        for noise_cov in [None, 10 ** -6, 0.01, 10]
    ],
    name="symmlin_belief_update",
)
def fixture_symmlin_belief_update(
    request,
    n: int,
    linsys_spd: LinearSystem,
    symm_belief: beliefs.SymmetricLinearSystemBelief,
    action: np.ndarray,
    matvec_observation: np.ndarray,
) -> belief_updates.SymmetricNormalLinearObsBeliefUpdate:
    """Belief update for the symmetric normal belief and linear observations."""
    return belief_updates.SymmetricNormalLinearObsBeliefUpdate(
        problem=linsys_spd,
        belief=symm_belief,
        actions=action,
        observations=matvec_observation,
        noise_cov=request.param,
    )


@pytest.fixture(name="weakmeancorrlin_belief_update")
def fixture_weakmeancorrlin_belief_update(
    n: int,
    linsys_spd: LinearSystem,
    weakmeancorr_belief: beliefs.WeakMeanCorrespondenceBelief,
    action: np.ndarray,
    matvec_observation: np.ndarray,
) -> belief_updates.WeakMeanCorrLinearObsBeliefUpdate:
    """Belief update for the weak mean correspondence belief and linear observations."""
    return belief_updates.WeakMeanCorrLinearObsBeliefUpdate(
        problem=linsys_spd,
        belief=weakmeancorr_belief,
        actions=action,
        observations=matvec_observation,
    )


#####################
# Stopping Criteria #
#####################


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
                stop_criteria.Residual(),
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
    """Observation operators of linear solvers."""
    return request.param


################################
# Probabilistic Linear Solvers #
################################


@pytest.fixture(name="solver_state_init")
def fixture_solver_state_init(
    linsys_spd: LinearSystem, prior: beliefs.LinearSystemBelief
) -> LinearSolverState:
    """Initial solver state of a probabilistic linear solver."""
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
        optimize_hyperparams=False,
        stopping_criteria=[stop_criteria.MaxIterations(), stop_criteria.Residual()],
    )
