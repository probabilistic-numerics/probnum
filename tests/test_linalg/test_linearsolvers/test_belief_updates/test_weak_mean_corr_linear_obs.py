"""Tests for the weak mean correspondence belief update under linear observations."""
import numpy as np
import pytest

from probnum.linalg.linearsolvers.belief_updates import (
    WeakMeanCorrLinearObsBeliefUpdate,
)

pytestmark = pytest.mark.usefixtures("weakmeancorrlin_belief_update")


def posterior_params(action, observation, prior_mean, prior_cov_factor, unc_scale=1.0):
    """Posterior parameters of the symmetric linear Gaussian model."""
    delta = observation - prior_mean @ action
    u = prior_cov_factor @ action / (action.T @ prior_cov_factor @ action).item()
    posterior_mean = prior_mean + delta @ u.T + u @ delta.T - u @ action.T @ delta @ u.T
    prior_cov_factor = prior_mean @ action @ action.T @ prior_mean / (
        action.T @ prior_mean @ action
    ).item() + unc_scale * (
        np.eye(prior_mean.shape[0]) - action @ action.T / (action.T @ action).item()
    )
    posterior_cov_factor = prior_cov_factor - prior_cov_factor @ action @ u.T
    return posterior_mean, posterior_cov_factor


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
#             phi = abs(.rng.normal())
#             psi = 1 / phi
#             s = .rng.normal(size=(n, 1))
#             y = linsys.A @ s
#
#             A1, V1 = .posterior_params(
#                 action=s,
#                 observation=y,
#                 prior_mean=A0,
#                 prior_cov_factor=linsys.A,
#                 unc_scale=phi,
#             )
#             Ainv1, W1 = .posterior_params(
#                 action=y,
#                 observation=s,
#                 prior_mean=Ainv0,
#                 prior_cov_factor=Ainv0,
#                 unc_scale=psi,
#             )
#
#             # Computation via belief update
#             prior = WeakMeanCorrespondenceBelief(
#                 A0=A0,
#                 Ainv0=Ainv0,
#                 b=linsys.b,
#                 phi=phi,
#                 psi=psi,
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


def test_means_correspond_weakly(
    weakmeancorrlin_belief_update: WeakMeanCorrLinearObsBeliefUpdate,
    matvec_observation: np.ndarray,
):
    r"""Test whether :math:`\mathbb{E}[A]^{-1}y = \mathbb{E}[H]y` for all actions
    :math:`y`."""
    np.testing.assert_allclose(
        np.linalg.solve(
            weakmeancorrlin_belief_update.A.mean.todense(), matvec_observation
        ),
        weakmeancorrlin_belief_update.Ainv.mean @ matvec_observation,
    )


@pytest.mark.parametrize("n", [3, 5, 10], indirect=True)
def test_iterative_covariance_trace_update(
    n: int,
    weakmeancorrlin_belief_update: WeakMeanCorrLinearObsBeliefUpdate,
):
    """The solver's returned value for the trace must match the actual trace of the
    solution covariance."""
    pytest.approx(
        weakmeancorrlin_belief_update.A.cov.trace(),
        np.trace(weakmeancorrlin_belief_update.A.cov.todense()),
    )
    pytest.approx(
        weakmeancorrlin_belief_update.Ainv.cov.trace(),
        np.trace(weakmeancorrlin_belief_update.Ainv.cov.todense()),
    )
