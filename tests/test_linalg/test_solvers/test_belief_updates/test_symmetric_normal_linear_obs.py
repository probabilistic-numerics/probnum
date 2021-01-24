"""Test case for the symmetric normal belief update under linear observations."""
from typing import Union

import numpy as np
import pytest

import probnum.linops as linops
from probnum.linalg.solvers.belief_updates import SymmetricNormalLinearObsBeliefUpdate
from probnum.linalg.solvers.beliefs import LinearSystemBelief
from probnum.problems import LinearSystem


def posterior_params(
    action_vec: np.ndarray,
    observation_vec: np.ndarray,
    prior_mean: Union[np.ndarray, linops.LinearOperator],
    prior_cov_factor: Union[np.ndarray, linops.LinearOperator],
):
    """Posterior parameters of the symmetric linear Gaussian model."""
    delta = observation_vec - prior_mean @ action_vec
    u = prior_cov_factor @ action_vec / (action_vec.T @ (prior_cov_factor @ action_vec))
    posterior_mean = (
        linops.aslinop(prior_mean).todense()
        + delta @ u.T
        + u @ delta.T
        - u @ action_vec.T @ delta @ u.T
    )
    posterior_cov_factor = (
        linops.aslinop(prior_cov_factor).todense() - prior_cov_factor @ action_vec @ u.T
    )
    return posterior_mean, posterior_cov_factor


def test_symmetric_posterior_params(
    symlin_updated_belief: LinearSystemBelief,
):
    """Test whether posterior parameters are symmetric."""

    for linop in [
        symlin_updated_belief.A.mean,
        symlin_updated_belief.A.cov.A,
        symlin_updated_belief.Ainv.mean,
        symlin_updated_belief.Ainv.cov.A,
    ]:
        mat = linop.todense()
        np.testing.assert_allclose(mat, mat.T, rtol=10 ** 6 * np.finfo(float).eps)


def test_matrix_posterior_computation(
    n: int,
    linsys_spd: LinearSystem,
    action: np.ndarray,
    matvec_observation: np.ndarray,
    linobs_belief_update: SymmetricNormalLinearObsBeliefUpdate,
):
    """Test the posterior computation of the belief update against the theoretical
    expressions."""

    belief = linobs_belief_update.belief

    # Posterior mean and covariance factor
    A_mean_updated, A_covfactor_updated = posterior_params(
        action=action,
        observation=matvec_observation,
        prior_mean=belief.A.mean,
        prior_cov_factor=belief.A.cov.A,
    )
    Ainv_mean_updated, Ainv_covfactor_updated = posterior_params(
        action=matvec_observation,
        observation=action,
        prior_mean=belief.Ainv.mean,
        prior_cov_factor=belief.Ainv.cov.A,
    )

    np.testing.assert_allclose(
        A_mean_updated,
        linobs_belief_update.A.mean.todense(),
        err_msg="The posterior mean for A does not match its definition.",
    )
    np.testing.assert_allclose(
        A_covfactor_updated,
        linobs_belief_update.A.cov.A.todense(),
        err_msg="The posterior covariance factor for A does not match its "
        "definition.",
    )

    np.testing.assert_allclose(
        Ainv_mean_updated,
        linobs_belief_update.Ainv.mean.todense(),
        err_msg="The posterior mean for Ainv does not match its definition.",
    )
    np.testing.assert_allclose(
        Ainv_covfactor_updated,
        linobs_belief_update.Ainv.cov.A.todense(),
        err_msg="The posterior covariance factor for Ainv does not match its "
        "definition.",
    )


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
