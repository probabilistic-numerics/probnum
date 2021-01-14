"""Tests for probabilistic linear solvers."""

import numpy as np
import pytest

from probnum import random_variables as rvs
from probnum.linalg import problinsolve

LINSOLVE_RELTOL = 10 ** -6
LINSOLVE_ABSTOL = 10 ** -6

# pylint: disable="invalid-name"


def test_prior_information(linsolve):
    """The solver should automatically handle different types of prior information."""
    # TODO
    pass


def test_prior_dimension_mismatch(linsolve):
    """Test whether the probabilistic linear solver throws an exception for priors with
    mismatched dimensions."""
    A = np.zeros(shape=[3, 3])
    with pytest.raises(
        ValueError, msg="Invalid input formats should raise a ValueError."
    ):
        # A inverse not square
        problinsolve(
            A=A,
            b=np.zeros(A.shape[0]),
            Ainv0=np.zeros([2, 3]),
            x0=np.zeros(shape=[A.shape[1]]),
        )
        # A, Ainv dimension mismatch
        problinsolve(
            A=A,
            b=np.zeros(A.shape[0]),
            Ainv0=np.zeros([2, 2]),
            x0=np.zeros(shape=[A.shape[1]]),
        )


def test_system_dimension_mismatch(linsolve):
    """Test whether linear solvers throw an exception for input with mismatched
    dimensions."""
    A = np.zeros(shape=[3, 3])
    b = np.zeros(shape=[4])
    x0 = np.zeros(shape=[1])
    with pytest.raises(
        ValueError, msg="Invalid input formats should raise a ValueError."
    ):
        # A, b dimension mismatch
        linsolve(A=A, b=b)
        # A, x0 dimension mismatch
        linsolve(A=A, b=np.zeros(A.shape[0]), x0=x0)
        # A not square
        linsolve(
            A=np.zeros([3, 4]),
            b=np.zeros(A.shape[0]),
            x0=np.zeros(shape=[A.shape[1]]),
        )


def test_randvar_output(linsys_spd, linsolve):
    """Probabilistic linear solvers output random variables."""
    x, A, Ainv, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b)
    for rv in [x, A, Ainv]:
        assert isinstance(
            rv, rvs.RandomVariable
        ), "Output of probabilistic linear solver is not a random variable."


def test_posterior_means_symmetric(linsys_spd, linsolve):
    """Test whether the posterior means of the matrix models are symmetric."""
    _, A, Ainv, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b)
    np.testing.assert_allclose(A.mean, A.mean.T)
    np.testing.assert_allclose(Ainv.mean, Ainv.mean.T)


def test_posterior_means_positive_definite(linsys_spd, linsolve):
    """Test whether the posterior means of the matrix models are positive definite."""
    _, A, Ainv, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b)
    assert np.all(np.eigh(A.mean) >= 0.0)
    assert np.all(np.eigh(Ainv.mean) >= 0.0)


def test_zero_rhs(A_spd, linsolve):
    """Linear system with zero right hand side."""
    b = np.zeros(A_spd.shape[0])
    tols = np.r_[np.logspace(np.log10(1e-10), np.log10(1e2), 7)]

    for tol in tols:
        x, _, _, _ = linsolve(A=A_spd, b=b, atol=tol)
        np.testing.assert_allclose(x.mean, 0, atol=np.finfo(float).eps)


def test_multiple_rhs(A_spd, linsolve):
    """Linear system with matrix right hand side."""
    B = np.random.rand(A_spd.shape[0], 5)

    x, _, _, info = linsolve(A=A_spd, b=B)
    assert (x.shape == B.shape, "Shape of solution and right hand side do not match.")
    np.testing.assert_allclose(x.mean, np.linalg.solve(A_spd, B))


def test_spd_system(linsys_spd, linsolve):
    """Random symmetric positive definite linear system."""
    x, _, _, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_spd.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match true solution.",
    )


def test_sparse_spd_system(linsys_sparse_spd, linsolve):
    """Random sparse symmetric positive definite linear system."""
    x, _, _, _ = linsolve(A=linsys_sparse_spd.A, b=linsys_sparse_spd.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_sparse_spd.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match true solution.",
    )


def test_sparse_poisson_system(linsys_poisson, linsolve):
    """(Sparse) linear system from Poisson PDE with boundary conditions."""
    x, _, _, _ = linsolve(A=linsys_poisson.A, b=linsys_poisson.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_poisson.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match solution from scipy.sparse.linalg.spsolve.",
    )


def test_kernel_matrix_system(linsys_kernel, linsolve):
    """Linear system with a kernel matrix."""
    x, _, _, _ = linsolve(A=linsys_kernel.A, b=linsys_kernel.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_kernel.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match true solution.",
    )
