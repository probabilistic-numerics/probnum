"""Tests for probabilistic linear solvers."""

from typing import Callable

import numpy as np
import pytest

from probnum import linops, randvars
from probnum.problems import LinearSystem
from probnum.type import MatrixArgType

LINSOLVE_RELTOL = 10 ** -5
LINSOLVE_ABSTOL = 10 ** -5

# pylint: disable="invalid-name"


def test_preconditioner(
    linsys_spd: LinearSystem, preconditioner: MatrixArgType, linsolve: Callable
):
    """The solver should be able to take a preconditioner."""
    x, _, _, _, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b, Ainv0=preconditioner)
    np.testing.assert_allclose(
        x.mean, linsys_spd.solution, atol=LINSOLVE_ABSTOL, rtol=LINSOLVE_RELTOL
    )


def test_prior_dimension_mismatch(linsolve: Callable):
    """Test whether the probabilistic linear solver throws an exception for priors with
    mismatched dimensions."""
    A = np.zeros(shape=[3, 3])
    with pytest.raises(ValueError):
        # A, Ainv dimension mismatch
        linsolve(
            A=A,
            b=np.zeros(A.shape[0]),
            Ainv0=linops.aslinop(np.eye(2)),
            x0=np.zeros(shape=[A.shape[1]]),
        )


def test_system_dimension_mismatch(linsolve: Callable):
    """Test whether linear solvers throw an exception for input with mismatched
    dimensions."""
    A = np.zeros(shape=[3, 3])
    b = np.zeros(shape=[4])
    x0 = np.zeros(shape=[1])
    with pytest.raises(ValueError):
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


def test_randvar_output(linsys_spd: LinearSystem, linsolve: Callable):
    """Probabilistic linear solvers output random variables."""
    x, A, Ainv, b, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b)
    for rv in [x, A, Ainv, b]:
        assert isinstance(
            rv, randvars.RandomVariable
        ), "Output of probabilistic linear solver is not a random variable."


def test_posterior_means_symmetric(linsys_spd: LinearSystem, linsolve: Callable):
    """Test whether the posterior means of the matrix models are symmetric."""
    _, A, Ainv, _, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b)
    np.testing.assert_allclose(A.mean.todense(), A.mean.T.todense())
    np.testing.assert_allclose(Ainv.mean.todense(), Ainv.mean.T.todense())


def test_posterior_means_positive_definite(
    linsys_spd: LinearSystem, linsolve: Callable
):
    """Test whether the posterior means of the matrix models are positive definite."""
    _, A, Ainv, _, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b)
    assert np.all(np.linalg.eigh(A.mean.todense())[0] >= 0.0)
    assert np.all(np.linalg.eigh(Ainv.mean.todense())[0] >= 0.0)


def test_zero_rhs(spd_mat: np.ndarray, linsolve: Callable):
    """Linear system with zero right hand side."""
    b = np.zeros(spd_mat.shape[0])
    tols = np.r_[np.logspace(np.log10(1e-10), np.log10(1e2), 7)]

    for tol in tols:
        x, _, _, _, _ = linsolve(A=spd_mat, b=b, atol=tol)
        np.testing.assert_allclose(x.mean, 0, atol=np.finfo(float).eps)


@pytest.mark.xfail(
    reason="The induced belief on x is not yet implemented for multiple rhs. "
    "github #302",
)
def test_multiple_rhs(linsys_spd_multiple_rhs: LinearSystem, linsolve: Callable):
    """Linear system with matrix right hand side."""
    x, _, _, _, _ = linsolve(A=linsys_spd_multiple_rhs.A, b=linsys_spd_multiple_rhs.b)
    assert (
        x.shape == linsys_spd_multiple_rhs.b.shape
    ), "Shape of solution and right hand side do not match."
    np.testing.assert_allclose(x.mean, linsys_spd_multiple_rhs.solution)


def test_spd_system(linsys_spd: LinearSystem, linsolve: Callable):
    """Random symmetric positive definite linear system."""
    x, _, _, _, _ = linsolve(A=linsys_spd.A, b=linsys_spd.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_spd.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match true solution.",
    )


def test_sparse_spd_system(linsys_sparse_spd: LinearSystem, linsolve: Callable):
    """Random sparse symmetric positive definite linear system."""
    x, _, _, _, _ = linsolve(A=linsys_sparse_spd.A, b=linsys_sparse_spd.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_sparse_spd.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match true solution.",
    )


def test_sparse_poisson_system(linsys_poisson: LinearSystem, linsolve: Callable):
    """(Sparse) linear system from Poisson PDE with boundary conditions."""
    x, _, _, _, _ = linsolve(A=linsys_poisson.A, b=linsys_poisson.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_poisson.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match solution from scipy.sparse.linalg.spsolve.",
    )


def test_kernel_matrix_system(linsys_kernel: LinearSystem, linsolve: Callable):
    """Linear system with a kernel matrix."""
    x, _, _, _, _ = linsolve(A=linsys_kernel.A, b=linsys_kernel.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_kernel.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match true solution.",
    )


def test_iid_noise_system(linsys_iid_noise: LinearSystem, linsolve: Callable):
    """Linear system with additive noise on the system matrix and right hand side."""
    x, _, _, _, _ = linsolve(A=linsys_iid_noise.A, b=linsys_iid_noise.b)
    np.testing.assert_allclose(
        x.mean,
        linsys_iid_noise.solution,
        rtol=LINSOLVE_RELTOL,
        atol=LINSOLVE_ABSTOL,
        err_msg="Solution does not match true solution.",
    )
