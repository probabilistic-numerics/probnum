"""Test fixtures for the belief update of probabilistic linear solvers."""


import numpy as np
import pytest

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.solvers import belief_updates, beliefs, hyperparams
from probnum.linalg.solvers.data import (
    LinearSolverAction,
    LinearSolverData,
    LinearSolverObservation,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix


@pytest.fixture(
    params=[
        pytest.param(bel_upd, id=bel_upd[0])
        for bel_upd in [
            (
                "symlin",
                beliefs.SymmetricNormalLinearSystemBelief,
                belief_updates.SymmetricNormalLinearObsBeliefUpdate,
            ),
            (
                "weakmeancorrlin",
                beliefs.WeakMeanCorrespondenceBelief,
                belief_updates.WeakMeanCorrLinearObsBeliefUpdate,
            ),
        ]
    ],
    name="symlin_updated_belief",
)
def fixture_symlin_updated_belief(
    request,
    n: int,
    random_state: np.random.RandomState,
    linsys_spd: LinearSystem,
    action: LinearSolverAction,
    matvec_observation: LinearSolverData,
) -> beliefs.LinearSystemBelief:
    """Belief update for a Gaussian prior and linear observations."""
    belief = request.param[1].from_inverse(
        linops.MatrixMult(random_spd_matrix(dim=n, random_state=random_state)),
        problem=linsys_spd,
    )
    belief_update = request.param[2](problem=linsys_spd, prior=belief)

    return belief_update(
        belief=belief,
        action=action,
        observation=matvec_observation,
    )[0]


@pytest.fixture(
    params=[
        pytest.param(noise_cov, id=f"noise{noise_cov}")
        for noise_cov in [None, 10 ** -6, 0.01, 10]
    ],
    name="noisy_updated_belief",
)
def fixture_noisy_updated_belief(
    request,
    n: int,
    linsys_spd: LinearSystem,
    symm_belief: beliefs.SymmetricNormalLinearSystemBelief,
    action: LinearSolverAction,
    matvec_observation: LinearSolverObservation,
) -> beliefs.LinearSystemBelief:
    """Belief update for the symmetric normal belief and linear observations."""
    noise = hyperparams.LinearSystemNoise(
        A_eps=rvs.Normal(
            mean=np.zeros((n, 1)),
            cov=linops.SymmetricKronecker(
                A=linops.ScalarMult(shape=(n, n), scalar=request.param)
            ),
        )
    )

    return belief_updates.SymmetricNormalLinearObsBeliefUpdate(
        problem=linsys_spd,
        prior=symm_belief,
    )(
        belief=symm_belief,
        action=action,
        observation=matvec_observation,
        hyperparams=noise,
    )[
        0
    ]


@pytest.fixture(name="weakmeancorr_updated_belief")
def fixture_weakmeancorr_updated_belief(
    n: int,
    linsys_spd: LinearSystem,
    weakmeancorr_belief: beliefs.WeakMeanCorrespondenceBelief,
    action: LinearSolverAction,
    matvec_observation: LinearSolverObservation,
) -> beliefs.LinearSystemBelief:
    """Belief update for the weak mean correspondence belief and linear observations."""
    return belief_updates.WeakMeanCorrLinearObsBeliefUpdate(
        problem=linsys_spd,
        prior=weakmeancorr_belief,
    )(belief=weakmeancorr_belief, action=action, observation=matvec_observation)[0]
