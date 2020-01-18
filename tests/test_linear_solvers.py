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
                               Ainv_mean.T, rtol=1e-4)
    np.testing.assert_allclose(Ainv_cov_A,
                               Ainv_cov_B, rtol=1e-6)
    np.testing.assert_allclose(Ainv_cov_A,
                               Ainv_cov_A.T, rtol=1e-4)


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


@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])  # , linear_solvers.bayescg])
def test_sparse_poisson(plinsolve):
    """Linear system with ill-conditioned matrix."""
    np.random.seed(1234)
    n = 40
    data = np.ones((3, n))
    data[0, :] = 2
    data[1, :] = -1
    data[2, :] = -1
    Poisson1D = scipy.sparse.spdiags(data, [0, -1, 1], n, n, format='csr')
    b = np.random.rand(n)
    # todo: use with pre-conditioner, as ill-conditioned for large n
    # print(np.linalg.cond(Poisson1D.todense()))
    # x = scipy.sparse.linalg.spsolve(Poisson1D, b)
    #
    # x_solver, _, _, info = plsolve(A=Poisson1D, b=b)
    # np.testing.assert_allclose(x_solver, x, rtol=1e-2)


@pytest.mark.parametrize("matlinsolve", [linear_solvers.problinsolve])
def test_matrixprior(matlinsolve):
    # Linear system
    n = 10
    A = np.random.rand(n, n)
    A = A.dot(A.T) + n * np.eye(n)  # Symmetrize and make diagonally dominant
    b = np.random.rand(n, 1)

    # Prior distribution on A
    covA = linear_operators.SymmetricKronecker(A=np.eye(n), B=np.eye(n))
    Ainv0 = probability.RandomVariable(distribution=probability.Normal(mean=np.eye(n), cov=covA))

    x, A, Ainv, info = matlinsolve(A=A, Ainv0=Ainv0, b=b)


@pytest.mark.parametrize("plinsolve", [linear_solvers.problinsolve])  # , linear_solvers.bayescg])
def test_searchdir_conjugacy(plinsolve):
    """Search directions should remain A-conjugate up to machine precision."""
    pass
