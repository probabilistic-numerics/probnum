import pytest
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from probnum.linalg import linear_solvers, linear_operators
from probnum import probability


# Linear solvers
@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve, linear_solvers.bayescg])
def test_dimension_mismatch(plinsolve):
    """Test whether linear solvers throw an exception for input with mismatched dimensions."""
    A = np.zeros(shape=[3, 3])
    b = np.zeros(shape=[4])
    x0 = np.zeros(shape=[1])
    assertion_warning = "Invalid input formats should raise a ValueError."
    with pytest.raises(ValueError) as e:
        # A, b dimension mismatch
        assert plinsolve(A=A, b=b), assertion_warning
        # A, x0 dimension mismatch
        assert plinsolve(A=A, b=np.zeros(A.shape[0]), x0=x0), assertion_warning
        # A not square
        assert plinsolve(A=np.zeros([3, 4]), b=np.zeros(A.shape[0]),
                         x0=np.zeros(shape=[A.shape[1]])), assertion_warning
        # A inverse not square
        assert plinsolve(A=A, b=np.zeros(A.shape[0]),
                         Ainv=np.zeros([2, 3]),
                         x0=np.zeros(shape=[A.shape[1]])), assertion_warning
        # A, Ainv dimension mismatch
        assert plinsolve(A=A, b=np.zeros(A.shape[0]),
                         Ainv=np.zeros([2, 2]),
                         x0=np.zeros(shape=[A.shape[1]])), assertion_warning


# todo: Write linear systems as parameters and test for output properties separately to run all combinations

@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])
def test_randvar_output(plinsolve):
    """Probabilistic linear solvers output random variables"""
    np.random.seed(1)
    n = 10
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T) + n * np.eye(n)
    b = np.random.rand(n)

    x, A, Ainv, _ = plinsolve(A=A, b=b)
    for rv in [x, A, Ainv]:
        assert isinstance(rv,
                          probability.RandomVariable), "Output of probabilistic linear solver is not a random variable."


@pytest.mark.parametrize("matblinsolve", [linear_solvers.problinsolve])
def test_symmetric_posterior_params(matblinsolve):
    """Test whether posterior parameters are symmetric."""
    np.random.seed(1)
    n = 10
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T) + n * np.eye(n)
    b = np.random.rand(n)

    _, _, Ainv, _ = matblinsolve(A=A, b=b)
    Ainv_mean = Ainv.mean().todense()
    Ainv_cov_A = Ainv.cov().A.todense()
    Ainv_cov_B = Ainv.cov().B.todense()
    np.testing.assert_allclose(Ainv_mean,
                               Ainv_mean.T, rtol=1e-6)
    np.testing.assert_allclose(Ainv_cov_A,
                               Ainv_cov_B, rtol=1e-6)
    np.testing.assert_allclose(Ainv_cov_A,
                               Ainv_cov_A.T, rtol=1e-6)


@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])  # , linear_solvers.bayescg])
def test_zero_rhs(plinsolve):
    """Linear system with zero right hand side."""
    np.random.seed(1234)
    A = np.random.rand(10, 10)
    A = A.dot(A.T) + 10 * np.eye(10)
    b = np.zeros(10)
    tols = np.r_[np.logspace(np.log10(1e-10), np.log10(1e2), 7)]

    for tol in tols:
        x, _, _, info = plinsolve(A=A, b=b, resid_tol=tol)
        np.testing.assert_allclose(x.mean(), 0, atol=1e-15)


@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])  # , linear_solvers.bayescg])
def test_multiple_rhs(plinsolve):
    """Linear system with matrix right hand side."""
    np.random.seed(42)
    A = np.random.rand(10, 10)
    A = A.dot(A.T) + 10 * np.eye(10)
    B = np.random.rand(10, 5)

    x, _, _, info = plinsolve(A=A, b=B)
    assert x.shape == B.shape, "Shape of solution and right hand side do not match."


@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])  # , linear_solvers.bayescg])
def test_spd_matrix(plinsolve):
    """Random spd matrix."""
    np.random.seed(1234)
    n = 40
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T) + n * np.eye(n)
    b = np.random.rand(n)
    x = np.linalg.solve(A, b)

    x_solver, _, _, info = plinsolve(A=A, b=b)
    np.testing.assert_allclose(x_solver.mean(), x, rtol=1e-4)


@pytest.fixture
def poisson_linear_system():
    """
    Poisson equation with Dirichlet conditions.

      - Laplace(u) = f    in the interior
                 u = u_D  on the boundary
    where
        u_D = 1 + x^2 + 2y^2
        f = -4

    Linear system resulting from discretization on an elliptic grid.
    """
    A = scipy.sparse.load_npz(file="resources/matrix_poisson.npz")
    f = np.load(file="resources/rhs_poisson.npy")
    return A, f


# TODO: run this test for a set of different linear systems
@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])  # , bayescg])
def test_sparse_poisson(plinsolve, poisson_linear_system):
    """(Sparse) linear system from Poisson PDE with boundary conditions."""
    A, f = poisson_linear_system
    u = scipy.sparse.linalg.spsolve(A=A, b=f)

    u_solver, Ahat, Ainvhat, info = plinsolve(A=A, b=f)
    np.testing.assert_allclose(u_solver.mean(), u, rtol=1e-5,
                               err_msg="Solution from probabilistic linear solver does" +
                                       " not match scipy.sparse.linalg.spsolve.")


@pytest.mark.parametrize("matblinsolve", [linear_solvers.problinsolve])
def test_solution_equivalence(matblinsolve, poisson_linear_system):
    """The induced distribution on the solution should match the estimated solution distribution: E[x] = E[A^-1] b"""
    A, f = poisson_linear_system

    # Solve linear system
    u_solver, Ahat, Ainvhat, info = matblinsolve(A=A, b=f)

    # E[x] = E[A^-1] b
    np.testing.assert_allclose(u_solver.mean(), (Ainvhat @ f[:, None]).mean().ravel(), rtol=1e-5,
                               err_msg="Solution from matrix-based probabilistic linear solver does not match the " +
                                       "estimated inverse, i.e. u =/= Ainv @ b ")


@pytest.mark.parametrize("matblinsolve", [linear_solvers.problinsolve])
def test_posterior_distribution_parameters(matblinsolve, poisson_linear_system):
    """Compute the posterior parameters of the matrix-based probabilistic linear solvers directly and compare."""
    # Initialization
    A, f = poisson_linear_system
    S = []  # search directions
    Y = []  # observations

    # Priors
    H0 = np.eye(A.shape[0])  # inverse prior mean
    A0 = np.eye(A.shape[0])  # prior mean
    WH0 = np.eye(A.shape[0])  # inverse prior Kronecker factor
    WA0 = np.eye(A.shape[0])  # prior Kronecker factor
    covH = linear_operators.SymmetricKronecker(WH0, WH0)
    covA = covH
    Ahat0 = probability.RandomVariable(distribution=probability.Normal(mean=A0, cov=covA))
    Ainvhat0 = probability.RandomVariable(distribution=probability.Normal(mean=H0, cov=covH))

    # Define callback function to obtain search directions
    def callback_postparams(xk, Ak, Ainvk, sk, yk, alphak, resid):
        S.append(sk)
        Y.append(yk)

    # Solve linear system
    u_solver, Ahat, Ainvhat, info = matblinsolve(A=A, b=f, A0=Ahat0, Ainv0=Ainvhat0, callback=callback_postparams)

    # Create arrays from lists
    S = np.squeeze(np.array(S)).T
    Y = np.squeeze(np.array(Y)).T

    # E[A] and E[A^-1]
    def posterior_mean(A0, WA0, S, Y):
        """Compute posterior mean of the symmetric probabilistic linear solver."""
        Delta = (Y - A0 @ S)
        U_T = np.linalg.solve(S.T @ WA0 @ S, S.T @ WA0)
        U = U_T.T
        Ak = A0 + Delta @ U_T + U @ Delta.T - U @ S.T @ Delta @ U_T
        return Ak

    Ak = posterior_mean(A0, WA0, S, Y)
    Hk = posterior_mean(H0, WH0, Y, S)

    np.testing.assert_allclose(Ahat.mean().todense(), Ak, rtol=1e-5,
                               err_msg="The matrix estimated by the probabilistic linear solver does not match the " +
                                       "directly computed one.")
    np.testing.assert_allclose(Ainvhat.mean().todense(), Hk, rtol=1e-5,
                               err_msg="The inverse matrix estimated by the probabilistic linear solver does not" +
                                       "match the directly computed one.")

    # Cov[A] and Cov[A^-1]
    def posterior_cov_kronfac(WA0, S):
        """Compute the covariance symmetric Kronecker factor of the probabilistic linear solver."""
        U_AT = np.linalg.solve(S.T @ WA0 @ S, S.T @ WA0)
        covfac = WA0 @ (np.identity(np.shape(WA0)[0]) - S @ U_AT)
        return covfac

    A_covfac = posterior_cov_kronfac(WA0, S)
    H_covfac = posterior_cov_kronfac(WH0, Y)

    np.testing.assert_allclose(Ahat.cov().A.todense(), A_covfac, rtol=1e-5,
                               err_msg="The covariance estimated by the probabilistic linear solver does not match the " +
                                       "directly computed one.")
    np.testing.assert_allclose(Ainvhat.cov().A.todense(), H_covfac, rtol=1e-5,
                               err_msg="The covariance estimated by the probabilistic linear solver does not" +
                                       "match the directly computed one.")


@pytest.mark.parametrize("matlinsolve", [linear_solvers.problinsolve])
def test_matrixprior(matlinsolve):
    """Solve random linear system with a matrix-based linear solver."""
    np.random.seed(1)
    # Linear system
    n = 10
    A = np.random.rand(n, n)
    A = A.dot(A.T) + n * np.eye(n)  # Symmetrize and make diagonally dominant
    b = np.random.rand(n, 1)

    # Prior distribution on A
    covA = linear_operators.SymmetricKronecker(A=np.eye(n), B=np.eye(n))
    Ainv0 = probability.RandomVariable(distribution=probability.Normal(mean=np.eye(n), cov=covA))

    x, Ahat, Ainvhat, info = matlinsolve(A=A, Ainv0=Ainv0, b=b)
    xnp = np.linalg.solve(A, b).ravel()

    np.testing.assert_allclose(x.mean(), xnp, rtol=1e-4,
                               err_msg="Solution does not match np.linalg.solve.")


@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])  # , linear_solvers.bayescg])
def test_searchdir_conjugacy(plinsolve, poisson_linear_system):
    """Search directions should remain A-conjugate up to machine precision, i.e. s_i^T A s_j = 0 for i != j."""
    searchdirs = []

    # Define callback function to obtain search directions
    def callback_searchdirs(xk, Ak, Ainvk, sk, yk, alphak, resid):
        searchdirs.append(sk)

    # Solve linear system
    A, f = poisson_linear_system
    plinsolve(A=A, b=f, callback=callback_searchdirs)

    # Compute pairwise inner products in A-space
    search_dir_arr = np.squeeze(np.array(searchdirs)).T
    inner_prods = search_dir_arr.T @ A @ search_dir_arr

    # Compare against identity matrix
    np.testing.assert_array_almost_equal(np.diag(np.diag(inner_prods)), inner_prods, decimal=6,
                                         err_msg="Search directions from solver are not A-conjugate.")


@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])  # , linear_solvers.bayescg])
def test_posterior_mean_CG_equivalency(plinsolve, poisson_linear_system):
    """The probabilistic linear solver should recover CG iterates as a posterior mean for specific covariances."""
    pass
