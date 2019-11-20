import pytest
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import probnum.linalg.linear_solvers


@pytest.mark.parametrize("prob_linear_solver",
                         [probnum.linalg.linear_solvers.MatrixBasedLinearSolver(),
                          probnum.linalg.linear_solvers.SolutionBasedConjugateGradients()])
def test_dimension_mismatch_exception(prob_linear_solver):
    """Test whether linear solvers throw an exception for input with mismatched dimensions."""
    A = np.zeros(shape=[3, 2])
    b = np.zeros(shape=[4])
    with pytest.raises(ValueError, match="Dimension mismatch.") as e:
        assert prob_linear_solver.solve(A=A, b=b), "Invalid input formats should raise a ValueError."


@pytest.mark.parametrize("prob_linear_solver",
                         [probnum.linalg.linear_solvers.MatrixBasedLinearSolver()])
def test_zero_rhs(prob_linear_solver):
    np.random.seed(1234)
    A = np.random.rand(10, 10)
    A = A.dot(A.T) + 10 * np.eye(10)
    b = np.zeros(10)
    tols = np.r_[np.logspace(np.log10(1e-10), np.log10(1e2), 7)]

    for tol in tols:
        x, info = prob_linear_solver.solve(A=A, b=b, resid_tol=tol)
        # np.testing.assert_equal(info, 0)
        np.testing.assert_allclose(x, 0, atol=1e-15)


@pytest.mark.parametrize("prob_linear_solver",
                         [probnum.linalg.linear_solvers.MatrixBasedLinearSolver()])
def test_spd_matrix(prob_linear_solver):
    np.random.seed(1234)
    n = 10
    A = np.random.rand(n, n)
    A = 0.5 * (A + A.T) + n * np.eye(n)
    b = np.random.rand(n)
    x = np.linalg.solve(A, b)

    x_solver, _ = prob_linear_solver.solve(A=A, b=b)
    np.testing.assert_allclose(x_solver, x, rtol=1e-2)


@pytest.mark.parametrize("prob_linear_solver",
                         [probnum.linalg.linear_solvers.MatrixBasedLinearSolver()])
def test_poisson(prob_linear_solver):
    np.random.seed(1234)
    n = 20
    data = np.ones((3, n))
    data[0, :] = 2
    data[1, :] = -1
    data[2, :] = -1
    Poisson1D = scipy.sparse.spdiags(data, [0, -1, 1], n, n, format='csr')
    b = np.random.rand(n)
    x = scipy.sparse.linalg.spsolve(Poisson1D, b)

    x_solver, _ = prob_linear_solver.solve(A=Poisson1D, b=b)
    np.testing.assert_allclose(x_solver, x, rtol=1e-2)
