"""Tests for the symmetric normal weak mean correspondence belief."""

import numpy as np
import pytest
import scipy.linalg

import probnum.linops as linops
from probnum.linalg.solvers.beliefs import WeakMeanCorrespondenceBelief
from probnum.linalg.solvers.data import LinearSolverData, LinearSolverObservation
from probnum.linalg.solvers.hyperparams import UncertaintyUnexploredSpace
from probnum.problems import LinearSystem

pytestmark = pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")


def test_means_correspond_weakly(
    weakmeancorr_belief: WeakMeanCorrespondenceBelief,
    solver_data: LinearSolverData,
    linsys_spd: LinearSystem,
):
    r"""Test whether :math:`\mathbb{E}[A]^{-1}y = \mathbb{E}[H]y` for all actions
    :math:`y`."""
    np.testing.assert_allclose(
        np.linalg.solve(
            weakmeancorr_belief.A.mean.todense(), solver_data.observations_arr.A
        ),
        weakmeancorr_belief.Ainv.mean @ solver_data.observations_arr.A,
    )


def test_system_matrix_uncertainty_in_action_span(
    weakmeancorr_belief: WeakMeanCorrespondenceBelief,
    solver_data: LinearSolverData,
    linsys_spd: LinearSystem,
    n: int,
):
    """Test whether the covariance factor W_0^A of the model for A acts like the
    true A in the span of the actions, i.e. if W_0^A S = Y."""
    np.testing.assert_allclose(
        solver_data.observations_arr.A,
        weakmeancorr_belief.A.cov.A @ solver_data.actions_arr.A,
    )


def test_inverse_uncertainty_in_observation_span(
    weakmeancorr_belief: WeakMeanCorrespondenceBelief,
    solver_data: LinearSolverData,
    linsys_spd: LinearSystem,
    n: int,
):
    """Test whether the covariance factor W_0^H of the model for Ainv acts like its
    prior mean in the span of the observations, i.e. if W_0^H Y = H_0 Y."""
    if n <= len(solver_data):
        pytest.skip("Action null space may be trivial.")

    np.testing.assert_allclose(
        weakmeancorr_belief.Ainv.mean @ solver_data.observations_arr.A,
        weakmeancorr_belief.Ainv.cov.A @ solver_data.observations_arr.A,
    )


@pytest.mark.parametrize("phi", [0, 10 ** -3, 1.0, 3.5])
def test_uncertainty_action_null_space_is_phi(
    phi: float,
    n: int,
    num_iters: int,
    solver_data: LinearSolverData,
    random_state: np.random.RandomState,
):
    r"""Test whether the uncertainty in the null space <S>^\perp is
    given by the uncertainty scale parameter phi for a scalar system matrix A."""
    if n <= len(solver_data):
        pytest.skip("Action null space may be trivial.")

    scalar_linsys = LinearSystem.from_matrix(
        A=linops.ScalarMult(scalar=2.5, shape=(n, n)), random_state=random_state
    )
    belief = WeakMeanCorrespondenceBelief(
        A0=scalar_linsys.A,
        Ainv0=scalar_linsys.A.inv(),
        b=scalar_linsys.b,
        uncertainty_scales=UncertaintyUnexploredSpace(
            Phi=phi, Psi=1 / phi if phi != 0.0 else 0.0
        ),
        data=LinearSolverData.from_arrays(
            actions_arr=solver_data.actions_arr,
            observations_arr=(
                scalar_linsys.A @ solver_data.actions_arr.A,
                np.repeat(scalar_linsys.b, num_iters, axis=1),
            ),
        ),
    )

    action_null_space = scipy.linalg.null_space(solver_data.actions_arr.A.T)

    np.testing.assert_allclose(
        action_null_space.T @ (belief.A.cov.A @ action_null_space),
        phi * np.eye(n - len(solver_data)),
        atol=10 ** -15,
        rtol=10 ** -15,
    )


@pytest.mark.parametrize("psi", [0, 10 ** -3, 1.0, 3.5 * 10 ** -5])
def test_uncertainty_observation_null_space_is_psi(
    psi: float,
    n: int,
    solver_data: LinearSolverData,
    random_state: np.random.RandomState,
):
    r"""Test whether the uncertainty in the null space <Y>^\perp is
    given by the uncertainty scale parameter psi for a scalar prior mean."""
    if n <= len(solver_data):
        pytest.skip("Observation null space may be trivial.")

    scalar_linsys = LinearSystem.from_matrix(
        A=linops.ScalarMult(scalar=2.5, shape=(n, n)), random_state=random_state
    )
    observations = scalar_linsys.A @ solver_data.actions_arr.A
    belief = WeakMeanCorrespondenceBelief(
        A0=scalar_linsys.A,
        Ainv0=scalar_linsys.A.inv(),
        b=scalar_linsys.b,
        uncertainty_scales=UncertaintyUnexploredSpace(
            Phi=1 / psi if psi != 0.0 else 0.0, Psi=psi
        ),
        data=solver_data,
    )

    observation_null_space = scipy.linalg.null_space(observations.T)

    np.testing.assert_allclose(
        observation_null_space.T @ (belief.Ainv.cov.A @ observation_null_space),
        psi * np.eye(n - len(solver_data)),
        atol=10 ** -15,
        rtol=10 ** -15,
    )


@pytest.mark.parametrize("phi,psi", [(1.0, 1.0), (0.0, 0.0), (10, 0.1)])
def test_no_data_prior(
    phi: float,
    psi: float,
    linsys: LinearSystem,
):
    """Test whether for no actions or observations the prior means and covariance are
    correct."""
    A0 = linsys.A
    Ainv0 = linops.Identity(shape=linsys.A.shape)
    belief = WeakMeanCorrespondenceBelief(
        A0=A0,
        Ainv0=Ainv0,
        b=linsys.b,
        uncertainty_scales=UncertaintyUnexploredSpace(Phi=phi, Psi=psi),
    )
    # Means
    if isinstance(A0, scipy.sparse.spmatrix):
        assert (belief.A.mean != A0).nnz == 0
    else:
        assert np.all(belief.A.mean == A0)

    if isinstance(Ainv0, scipy.sparse.spmatrix):
        assert (belief.Ainv.mean != Ainv0).nnz == 0
    else:
        assert np.all(belief.Ainv.mean == Ainv0)

    # Covariances
    assert isinstance(belief.A.cov.A, linops.ScalarMult)
    assert belief.A.cov.A.scalar == phi

    assert isinstance(belief.Ainv.cov.A, linops.ScalarMult)
    assert belief.Ainv.cov.A.scalar == psi


def test_inverse_nonscalar_prior_mean(
    n: int,
    weakmeancorr_belief: WeakMeanCorrespondenceBelief,
    solver_data: LinearSolverData,
    linsys_spd: LinearSystem,
):
    """Test whether the covariance for the inverse model with a non-scalar prior mean
    matches a naively computed one."""
    W0_Ainv = weakmeancorr_belief.Ainv0 @ linops.OrthogonalProjection(
        subspace_basis=solver_data.observations_arr.A,
        innerprod_matrix=weakmeancorr_belief.Ainv0,
    ).todense() + (
        np.eye(n)
        - linops.OrthogonalProjection(
            subspace_basis=solver_data.observations_arr.A
        ).todense()
    )

    np.testing.assert_allclose(
        weakmeancorr_belief.Ainv.cov.A.todense(),
        W0_Ainv,
        err_msg="Covariance factor of the Ainv model does not match "
        "naively computed one.",
        atol=10 ** 2 * np.finfo(float).eps,
    )


@pytest.mark.parametrize("alpha", [0.5, 1.0, 10.0])
@pytest.mark.parametrize("phi, psi", [(0.2, 1.2), (10 ** -4, 10 ** 4), (0.0, 0.0)])
def test_conjugate_actions_covariance(
    alpha: float,
    phi: float,
    psi: float,
    n: int,
    solver_data: LinearSolverData,
    linsys_spd: LinearSystem,
):
    """Test whether the covariance for conjugate actions matches a naively computed
    one."""
    # Compute conjugate actions via Cholesky decomposition: S' = L^{-T}S
    orth_actions = scipy.linalg.orth(solver_data.actions_arr.A)
    chol = scipy.linalg.cholesky(linsys_spd.A, lower=False)
    conj_actions = scipy.linalg.solve_triangular(chol, orth_actions, lower=False)
    observations = linsys_spd.A @ conj_actions

    # Inverse prior mean
    Ainv0 = linops.ScalarMult(scalar=alpha, shape=(n, n))

    # Naive covariance factors
    W0_A = observations @ np.linalg.solve(
        conj_actions.T @ observations, observations.T
    ) + phi * (
        np.eye(linsys_spd.A.shape[0])
        - linops.OrthogonalProjection(subspace_basis=conj_actions).todense()
    )
    W0_Ainv = (
        psi * np.eye(linsys_spd.A.shape[0])
        + (Ainv0.scalar - psi)
        * linops.OrthogonalProjection(subspace_basis=observations).todense()
    )

    belief = WeakMeanCorrespondenceBelief(
        A0=linsys_spd.A,
        Ainv0=Ainv0,
        b=linsys_spd.b,
        uncertainty_scales=UncertaintyUnexploredSpace(Phi=phi, Psi=psi),
        data=LinearSolverData.from_arrays(
            actions_arr=(conj_actions, None), observations_arr=(observations, None)
        ),
        action_obs_innerprods=np.einsum("nk,nk->k", conj_actions, observations),
    )

    np.testing.assert_allclose(
        belief.A.cov.A.todense(),
        W0_A,
        err_msg="Covariance factor of the A model does not match "
        "naively computed one.",
    )
    np.testing.assert_allclose(
        belief.Ainv.cov.A.todense(),
        W0_Ainv,
        err_msg="Covariance factor of the Ainv model does not match "
        "naively computed one.",
    )


# Classmethod tests
def test_from_matrix_satisfies_mean_correspondence(linsys: LinearSystem):
    """Test whether for a belief constructed from an approximate system matrix, the
    prior mean of the inverse model corresponds."""
    A0 = linops.ScalarMult(scalar=5.0, shape=linsys.A.shape)
    belief = WeakMeanCorrespondenceBelief.from_matrix(A0=A0, problem=linsys)
    np.testing.assert_allclose(
        belief.Ainv.mean.inv().todense(), belief.A.mean.todense()
    )


def test_from_inverse_satisfies_mean_correspondence(linsys: LinearSystem):
    """Test whether for a belief constructed from an approximate inverse, the prior mean
    of the system matrix model corresponds."""
    Ainv0 = linops.ScalarMult(scalar=5.0, shape=linsys.A.shape)
    belief = WeakMeanCorrespondenceBelief.from_inverse(Ainv0=Ainv0, problem=linsys)
    np.testing.assert_allclose(
        belief.Ainv.mean.inv().todense(), belief.A.mean.todense()
    )


@pytest.mark.parametrize("alpha", [10 ** -16, 1.0, 10])
def test_from_scalar(alpha: float, linsys: LinearSystem):
    """Test whether a linear system belief can be created from a scalar."""
    WeakMeanCorrespondenceBelief.from_scalar(scalar=alpha, problem=linsys)


@pytest.mark.parametrize("alpha", [-1.0, -10, 0.0, 0])
def test_from_scalar_nonpositive_raises_value_error(alpha: float, linsys: LinearSystem):
    """Test whether attempting to construct a weak mean correspondence belief from a
    non-positive scalar results in a ValueError."""
    with pytest.raises(ValueError):
        WeakMeanCorrespondenceBelief.from_scalar(scalar=alpha, problem=linsys)
