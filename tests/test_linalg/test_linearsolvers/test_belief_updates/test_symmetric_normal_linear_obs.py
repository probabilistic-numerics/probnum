"""Test case for the symmetric normal belief update under linear observations."""
from typing import Union

import numpy as np
import pytest

import probnum.linops as linops
from probnum.linalg.linearsolvers.belief_updates import (
    SymmetricNormalLinearObsBeliefUpdate,
)

pytestmark = pytest.mark.usefixtures("symmlin_belief_update")


def posterior_params(
    action: np.ndarray,
    observation: np.ndarray,
    prior_mean: Union[np.ndarray, linops.LinearOperator],
    prior_cov_factor: Union[np.ndarray, linops.LinearOperator],
):
    """Posterior parameters of the symmetric linear Gaussian model."""
    delta = observation - prior_mean @ action
    u = prior_cov_factor @ action / (action.T @ prior_cov_factor @ action)
    posterior_mean = prior_mean + delta @ u.T + u @ delta.T - u @ action.T @ delta @ u.T
    posterior_cov_factor = prior_cov_factor - prior_cov_factor @ action @ u.T
    return posterior_mean, posterior_cov_factor


def test_symmetric_posterior_params(
    symmlin_belief_update: SymmetricNormalLinearObsBeliefUpdate,
):
    """Test whether posterior parameters are symmetric."""
    A = symmlin_belief_update.A
    Ainv = symmlin_belief_update.Ainv

    for linop in [A.mean, A.cov.A, Ainv.mean, Ainv.cov.A]:
        mat = linop.todense()
        np.testing.assert_allclose(mat, mat.T, rtol=10 ** 6 * np.finfo(float).eps)


# def test_matrix_posterior_computation():
#     """Test the posterior computation of the belief update against the theoretical
#     expressions."""
#     # pylint : disable="too-many-locals"
#     for n in [10, 50, 100]:
#         with .subTest():
#             A = random_spd_matrix(dim=n, random_state=.rng)
#             b = .rng.normal(size=(n, 1))
#             linsys = LinearSystem(A, b)
#
#             # Posterior mean and covariance factor
#             A0 = random_spd_matrix(dim=n, random_state=.rng)
#             Ainv0 = random_spd_matrix(dim=n, random_state=.rng)
#             V0 = random_spd_matrix(dim=n, random_state=.rng)
#             W0 = random_spd_matrix(dim=n, random_state=.rng)
#             prior_A = rvs.Normal(mean=A0, cov=linops.SymmetricKronecker(V0))
#             prior_Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(W0))
#             s = .rng.normal(size=(n, 1))
#             y = linsys.A @ s
#
#             A1, V1 = .posterior_params(
#                 action=s, observation=y, prior_mean=A0, prior_cov_factor=V0
#             )
#             Ainv1, W1 = .posterior_params(
#                 action=y, observation=s, prior_mean=Ainv0, prior_cov_factor=W0
#             )
#
#             # Computation via belief update
#             prior = LinearSystemBelief(
#                 x=prior_Ainv @ b,
#                 A=prior_A,
#                 Ainv=prior_Ainv,
#                 b=rvs.Constant(linsys.b),
#             )
#             prior.update(
#                 problem=linsys,
#                 observation_op=MatVecObservation(),
#                 action=s,
#                 observation=y,
#             )
#
#             .assertAllClose(
#                 A1,
#                 prior.A.mean.todense(),
#                 msg="The posterior mean for A does not match its definition.",
#             )
#             .assertAllClose(
#                 V1,
#                 prior.A.cov.A.todense(),
#                 msg="The posterior covariance factor for A does not match its "
#                 "definition.",
#             )
#
#             .assertAllClose(
#                 Ainv1,
#                 prior.Ainv.mean.todense(),
#                 msg="The posterior mean for Ainv does not match its definition.",
#             )
#             .assertAllClose(
#                 W1,
#                 prior.Ainv.cov.A.todense(),
#                 msg="The posterior covariance factor for Ainv does not match its "
#                 "definition.",
#             )


def test_uncertainty_action_space_is_zero(
    symmlin_belief_update: SymmetricNormalLinearObsBeliefUpdate, action: np.ndarray
):
    """Test whether the uncertainty about the system matrix in the action span of the
    already explored directions is zero."""
    A = symmlin_belief_update.A
    np.testing.assert_allclose(
        np.zeros_like(action),
        A.cov.A @ action,
        atol=10 ** 3 * np.finfo(float).eps,
    )


def test_uncertainty_observation_space_is_zero(
    symmlin_belief_update: SymmetricNormalLinearObsBeliefUpdate,
    matvec_observation: np.ndarray,
):
    """Test whether the uncertainty about the inverse in the observation span of the
    already made observations is zero."""
    Ainv = symmlin_belief_update.Ainv
    np.testing.assert_allclose(
        np.zeros_like(matvec_observation),
        Ainv.cov.A @ matvec_observation,
        atol=10 ** 3 * np.finfo(float).eps,
    )
