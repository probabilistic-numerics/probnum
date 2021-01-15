"""Test cases for linear system beliefs."""

from typing import Union

import numpy as np
import pytest
import scipy.linalg

import probnum.linops as linops
import probnum.random_variables as rvs
from probnum.linalg.linearsolvers.beliefs import (
    LinearSystemBelief,
    SymmetricLinearSystemBelief,
    WeakMeanCorrespondenceBelief,
)
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_spd_matrix

# pylint: disable="invalid-name"


def test_dimension_mismatch_raises_value_error():
    """Test whether mismatched components result in a ValueError."""
    m, n, nrhs = 5, 3, 2
    A = rvs.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
    Ainv = A
    x = rvs.Normal(mean=np.zeros((n, nrhs)), cov=np.eye(n * nrhs))
    b = rvs.Constant(np.ones((m, nrhs)))

    # A does not match b
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=rvs.Constant(np.ones((m + 1, nrhs))))

    # A does not match x
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=Ainv,
            x=rvs.Normal(mean=np.zeros((n + 1, nrhs)), cov=np.eye((n + 1) * nrhs)),
            b=b,
        )

    # x does not match b
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=Ainv,
            x=rvs.Normal(mean=np.zeros((n, nrhs + 1)), cov=np.eye(n * (nrhs + 1))),
            b=b,
        )

    # A does not match Ainv
    with pytest.raises(ValueError):
        LinearSystemBelief(
            A=A,
            Ainv=rvs.Normal(mean=np.ones((m + 1, n)), cov=np.eye((m + 1) * n)),
            x=x,
            b=b,
        )


def test_beliefs_are_two_dimensional():
    """Check whether all beliefs over quantities of interest are two dimensional."""
    m, n = 5, 3
    A = rvs.Normal(mean=np.ones((m, n)), cov=np.eye(m * n))
    Ainv = A
    x = rvs.Normal(mean=np.zeros(n), cov=np.eye(n))
    b = rvs.Constant(np.ones(m))
    belief = LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=b)

    assert belief.A.ndim == 2
    assert belief.Ainv.ndim == 2
    assert belief.x.ndim == 2
    assert belief.b.ndim == 2


def test_non_two_dimensional_raises_value_error():
    """Test whether specifying higher-dimensional random variables raise a
    ValueError."""
    A = rvs.Constant(np.eye(5))
    Ainv = rvs.Constant(np.eye(5))
    x = rvs.Constant(np.ones((5, 1)))
    b = rvs.Constant(np.ones((5, 1)))

    # A.ndim == 3
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A[:, None], Ainv=Ainv, x=x, b=b)

    # Ainv.ndim == 3
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A, Ainv=Ainv[:, None], x=x, b=b)

    # x.ndim == 3
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A, Ainv=Ainv, x=x[:, None], b=b)

    # b.ndim == 3
    with pytest.raises(ValueError):
        LinearSystemBelief(A=A, Ainv=Ainv, x=x, b=b[:, None])


# Classmethod tests


def test_from_solution_array(
    belief_class: LinearSystemBelief,
    linsys: LinearSystem,
    random_state: np.random.RandomState,
):
    """Test whether a linear system belief can be created from a solution estimate given
    as an array."""
    x0 = random_state.normal(size=linsys.A.shape[1])
    belief_class.from_solution(x0=x0, problem=linsys)


def test_from_solution_generates_consistent_inverse_belief(
    belief_class: LinearSystemBelief,
    linsys: LinearSystem,
    random_state: np.random.RandomState,
):
    """Test whether the belief for the inverse generated from a solution guess matches
    the belief for the solution."""
    x0 = random_state.normal(size=linsys.A.shape[1])
    belief = belief_class.from_solution(x0=x0, problem=linsys)
    np.testing.assert_allclose(belief.x.mean, belief.Ainv.mean @ linsys.b)


def test_from_solution_creates_better_initialization(belief_class: LinearSystemBelief):
    """Test whether if possible, a better initial value x1 is constructed from x0."""
    x0_list = []
    linsys = LinearSystem(
        A=np.array([[4, 2, -6, 4], [2, 2, -3, 1], [-6, -3, 13, 0], [4, 1, 0, 30]]),
        solution=np.array([2, 0, -1, 2]),
        b=np.array([22, 9, -25, 68]),
    )

    # <b, x0> < 0
    x0_list.append(-linsys.b)

    # <b, x0> = 0, b != 0
    x0_list.append(np.array([0.5, -1, 0, -1 / 34])[:, None])
    pytest.approx((x0_list[1].T @ linsys.b).item(), 0.0)

    for x0 in x0_list:
        belief = belief_class.from_solution(x0=x0, problem=linsys)
        assert (
            (belief.x.mean.T @ linsys.b).item() > 0.0,
            "Inner product <x0, b>="
            f"{(belief.x.mean.T @ linsys.b).item():.4f} is not positive.",
        )
        error_x0 = ((linsys.solution - x0).T @ linsys.A @ (linsys.solution - x0)).item()
        error_x1 = (
            (linsys.solution - belief.x.mean).T
            @ linsys.A
            @ (linsys.solution - belief.x.mean)
        ).item()
        assert (
            error_x1 < error_x0,
            "Initialization for the solution x0 is not better in A-norm "
            "than the user-specified one.",
        )

    # b = 0
    linsys_homogeneous = LinearSystem(A=linsys.A, b=np.zeros_like(linsys.b))
    belief = belief_class.from_solution(
        x0=np.ones_like(linsys.b), problem=linsys_homogeneous
    )
    np.testing.assert_allclose(belief.x.mean, np.zeros_like(linsys.b))


def test_from_matrix(
    belief_class: LinearSystemBelief,
    mat: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    linsys: LinearSystem,
):
    """Test whether a linear system belief can be created from a system matrix estimate
    given as an array, sparse matrix or linear operator."""
    if (belief_class is WeakMeanCorrespondenceBelief) and not isinstance(
        mat, linops.LinearOperator
    ):
        with pytest.raises(TypeError):
            # Inefficient belief construction via explicit inversion raises error
            belief_class.from_matrix(A0=mat, problem=linsys)
    else:
        belief_class.from_matrix(A0=mat, problem=linsys)


def test_from_inverse(
    belief_class: LinearSystemBelief,
    mat: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    linsys: LinearSystem,
):
    """Test whether a linear system belief can be created from an inverse estimate given
    as an array, sparse matrix or linear operator."""
    if (belief_class is WeakMeanCorrespondenceBelief) and not isinstance(
        mat, linops.LinearOperator
    ):
        with pytest.raises(TypeError):
            belief_class.from_inverse(Ainv0=mat, problem=linsys)
    else:
        belief_class.from_inverse(Ainv0=mat, problem=linsys)


def test_from_matrices(
    belief_class: LinearSystemBelief,
    mat: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    linsys: LinearSystem,
):
    """Test whether a linear system belief can be created from an estimate of the system
    matrix and its inverse given as arrays, sparse matrices or linear operators."""
    belief_class.from_matrices(A0=mat, Ainv0=mat, problem=linsys)


class TestSymmetricLinearSystemBelief:
    """Tests for the symmetric Gaussian belief."""

    def test_induced_solution_has_correct_distribution(
        self, linsys_spd: LinearSystem, random_state: np.random.RandomState
    ):
        """Test whether the induced distribution over the solution from a belief over
        the inverse is correct."""
        Ainv0 = random_spd_matrix(dim=linsys_spd.A.shape[0], random_state=random_state)
        W = random_spd_matrix(dim=linsys_spd.A.shape[0], random_state=random_state)
        Ainv = rvs.Normal(mean=Ainv0, cov=linops.SymmetricKronecker(A=W))

        belief = SymmetricLinearSystemBelief.from_inverse(
            Ainv0=Ainv, problem=linsys_spd
        )

        np.testing.assert_allclose(
            belief.x.mean,
            belief.Ainv.mean @ linsys_spd.b,
            err_msg="Induced belief over the solution has an inconsistent mean.",
        )
        Wb = W @ linsys_spd.b
        bWb = (Wb.T @ linsys_spd.b).item()
        Sigma = 0.5 * (bWb * W + Wb @ Wb.T)
        np.testing.assert_allclose(
            belief.x.cov.todense(),
            Sigma,
            err_msg="Induced belief over the solution has an inconsistent covariance.",
        )


class TestWeakMeanCorrespondenceBelief:
    """Tests for the weak mean correspondence belief."""

    def test_means_correspond_weakly(
        self,
        weakmeancorr_belief: WeakMeanCorrespondenceBelief,
        actions: list,
        matvec_observations: list,
        linsys_spd: LinearSystem,
    ):
        r"""Test whether :math:`\mathbb{E}[A]^{-1}y = \mathbb{E}[H]y` for all actions
        :math:`y`."""
        np.testing.assert_allclose(
            np.linalg.solve(
                weakmeancorr_belief.A.mean.todense(), np.hstack(matvec_observations)
            ),
            weakmeancorr_belief.Ainv.mean @ np.hstack(matvec_observations),
        )

    def test_system_matrix_uncertainty_in_action_span(
        self,
        weakmeancorr_belief: WeakMeanCorrespondenceBelief,
        actions: list,
        matvec_observations: list,
        linsys_spd: LinearSystem,
    ):
        """Test whether the covariance factor W_0^A of the model for A acts like the
        true A in the span of the actions, i.e. if W_0^A S = Y."""
        np.testing.assert_allclose(
            np.hstack(matvec_observations),
            weakmeancorr_belief.A.cov.A @ np.hstack(actions),
        )

    def test_inverse_uncertainty_in_observation_span(
        self,
        weakmeancorr_belief: WeakMeanCorrespondenceBelief,
        actions: list,
        matvec_observations: list,
        linsys_spd: LinearSystem,
    ):
        """Test whether the covariance factor W_0^H of the model for Ainv acts like its
        prior mean in the span of the observations, i.e. if W_0^H Y = H_0 Y."""
        np.testing.assert_allclose(
            weakmeancorr_belief.Ainv.mean @ np.hstack(matvec_observations),
            weakmeancorr_belief.Ainv.cov.A @ np.hstack(matvec_observations),
        )

    @pytest.mark.parametrize("phi", [0, 10 ** -3, 1.0, 3.5])
    def test_uncertainty_action_null_space_is_phi(
        self,
        phi: float,
        n: int,
        actions: list,
        random_state: np.random.RandomState,
    ):
        r"""Test whether the uncertainty in the null space <S>^\perp is
        given by the uncertainty scale parameter phi for a scalar system matrix A."""
        if n <= len(actions):
            pytest.skip("Action null space may be trivial.")

        scalar_linsys = LinearSystem.from_matrix(
            A=linops.ScalarMult(scalar=2.5, shape=(n, n)), random_state=random_state
        )
        belief = WeakMeanCorrespondenceBelief(
            A0=scalar_linsys.A,
            Ainv0=scalar_linsys.A.inv(),
            b=scalar_linsys.b,
            phi=phi,
            psi=1 / phi if phi != 0.0 else 0.0,
            actions=actions,
            observations=scalar_linsys.A @ np.hstack(actions),
        )

        action_null_space = scipy.linalg.null_space(np.hstack(actions).T)

        np.testing.assert_allclose(
            action_null_space.T @ (belief.A.cov.A @ action_null_space),
            phi * np.eye(n - len(actions)),
            atol=10 ** -15,
            rtol=10 ** -15,
        )

    @pytest.mark.parametrize("psi", [0, 10 ** -3, 1.0, 3.5 * 10 ** -5])
    def test_uncertainty_observation_null_space_is_psi(
        self,
        psi: float,
        n: int,
        actions: list,
        random_state: np.random.RandomState,
    ):
        r"""Test whether the uncertainty in the null space <Y>^\perp is
        given by the uncertainty scale parameter psi for a scalar prior mean."""
        if n <= len(actions):
            pytest.skip("Observation null space may be trivial.")

        scalar_linsys = LinearSystem.from_matrix(
            A=linops.ScalarMult(scalar=2.5, shape=(n, n)), random_state=random_state
        )
        observations = scalar_linsys.A @ np.hstack(actions)
        belief = WeakMeanCorrespondenceBelief(
            A0=scalar_linsys.A,
            Ainv0=scalar_linsys.A.inv(),
            b=scalar_linsys.b,
            phi=1 / psi if psi != 0.0 else 0.0,
            psi=psi,
            actions=actions,
            observations=observations,
        )

        observation_null_space = scipy.linalg.null_space(observations.T)

        np.testing.assert_allclose(
            observation_null_space.T @ (belief.Ainv.cov.A @ observation_null_space),
            psi * np.eye(n - len(actions)),
            atol=10 ** -15,
            rtol=10 ** -15,
        )

    @pytest.mark.parametrize("phi,psi", [(1.0, 1.0), (0.0, 0.0), (10, 0.1)])
    def test_no_data_prior(
        self,
        phi: float,
        psi: float,
        linsys: LinearSystem,
    ):
        """Test whether for no actions or observations the prior means and covariance
        are correct."""
        A0 = linsys.A
        Ainv0 = linops.Identity(shape=linsys.A.shape)
        belief = WeakMeanCorrespondenceBelief(
            A0=A0,
            Ainv0=Ainv0,
            b=linsys.b,
            phi=phi,
            psi=psi,
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
        self,
        n: int,
        weakmeancorr_belief: WeakMeanCorrespondenceBelief,
        actions: list,
        matvec_observations: list,
        linsys_spd: LinearSystem,
    ):
        """Test whether the covariance for the inverse model with a non-scalar prior
        mean matches a naively computed one."""
        W0_Ainv = weakmeancorr_belief.Ainv0 @ linops.OrthogonalProjection(
            subspace_basis=np.hstack(matvec_observations),
            innerprod_matrix=weakmeancorr_belief.Ainv0,
        ).todense() + (
            np.eye(n)
            - linops.OrthogonalProjection(
                subspace_basis=np.hstack(matvec_observations)
            ).todense()
        )

        np.testing.assert_allclose(
            weakmeancorr_belief.Ainv.cov.A.todense(),
            W0_Ainv,
            err_msg="Covariance factor of the Ainv model does not match "
            "naively computed one.",
        )

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 10.0])
    @pytest.mark.parametrize("phi, psi", [(0.2, 1.2), (10 ** -4, 10 ** 4), (0.0, 0.0)])
    def test_conjugate_actions_covariance(
        self,
        alpha: float,
        phi: float,
        psi: float,
        n: int,
        actions: list,
        linsys_spd: LinearSystem,
    ):
        """Test whether the covariance for conjugate actions matches a naively computed
        one."""
        # Compute conjugate actions via Cholesky decomposition: S' = L^{-T}S
        actions = np.hstack(actions)
        chol = scipy.linalg.cholesky(linsys_spd.A)
        conj_actions = scipy.linalg.solve_triangular(chol, actions)
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
            phi=phi,
            psi=psi,
            actions=conj_actions,
            observations=observations,
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
    def test_from_matrix_satisfies_mean_correspondence(self, linsys: LinearSystem):
        """Test whether for a belief constructed from an approximate system matrix, the
        prior mean of the inverse model corresponds."""
        A0 = linops.ScalarMult(scalar=5.0, shape=linsys.A.shape)
        belief = WeakMeanCorrespondenceBelief.from_matrix(A0=A0, problem=linsys)
        np.testing.assert_allclose(
            belief.Ainv.mean.inv().todense(), belief.A.mean.todense()
        )

    def test_from_inverse_satisfies_mean_correspondence(self, linsys: LinearSystem):
        """Test whether for a belief constructed from an approximate inverse, the prior
        mean of the system matrix model corresponds."""
        Ainv0 = linops.ScalarMult(scalar=5.0, shape=linsys.A.shape)
        belief = WeakMeanCorrespondenceBelief.from_inverse(Ainv0=Ainv0, problem=linsys)
        np.testing.assert_allclose(
            belief.Ainv.mean.inv().todense(), belief.A.mean.todense()
        )

    @pytest.mark.parametrize("alpha", [10 ** -16, 1.0, 10])
    def test_from_scalar(self, alpha: float, linsys: LinearSystem):
        """Test whether a linear system belief can be created from a scalar."""
        WeakMeanCorrespondenceBelief.from_scalar(alpha=alpha, problem=linsys)

    @pytest.mark.parametrize("alpha", [-1.0, -10, 0.0, 0])
    def test_from_scalar_nonpositive_raises_value_error(
        self, alpha: float, linsys: LinearSystem
    ):
        """Test whether attempting to construct a weak mean correspondence belief from a
        non-positive scalar results in a ValueError."""
        with pytest.raises(ValueError):
            WeakMeanCorrespondenceBelief.from_scalar(alpha=alpha, problem=linsys)

    # Hyperparameters
    def test_uncertainty_calibration(self):
        """"""
        pass  # TODO

    def test_uncertainty_calibration_error(
        self, calibration_method: str, conj_grad_method
    ):
        """Test if the available uncertainty calibration procedures affect the error of
        the returned solution."""
        pass
        # tol = 10 ** -6
        #
        # x_est, Ahat, Ainvhat, info = conj_grad_method(
        #     A=A, b=b, calibration=calibration_method
        # )
        # assert(
        #     ((x_true - x_est.mean).T @ A @ (x_true - x_est.mean)).item() <= tol,
        #     "Estimated solution not sufficiently close to true solution.",
        # )
