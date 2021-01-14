"""Test fixtures for the linalg subpackage."""

import os
from typing import Callable

import numpy as np
import pytest
import scipy.sparse

import probnum
import probnum.kernels as kernels
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix


@pytest.fixture(params=[pytest.param(n, id=f"dim{n}") for n in [5, 10, 50, 100]])
def n(request) -> int:
    """Number of columns of the system matrix.

    This is mostly used for test parameterization.
    """
    return request.param


@pytest.fixture(params=[pytest.param(seed, id=f"seed{seed}") for seed in range(3)])
def random_state(request) -> np.random.RandomState:
    """Random states used to sample the test case input matrices.

    This is mostly used for test parameterization.
    """
    return np.random.RandomState(seed=request.param)


############
# Matrices #
############


@pytest.fixture()
def spd_mat(n: int, random_state: np.random.RandomState) -> np.ndarray:
    """Random symmetric positive definite matrix of dimension :func:`n`, sampled from
    :func:`random_state`."""
    return random_spd_matrix(dim=n, random_state=random_state)


@pytest.fixture(
    params=[
        pytest.param(sparsemat_density, id=f"density{sparsemat_density}")
        for sparsemat_density in (0.001, 0.01, 0.1)
    ]
)
def sparsemat_density(request):
    """Density of a sparse matrix defined by the fraction of nonzero entries."""
    return request.param


@pytest.fixture()
def sparse_spd_mat(
    sparsemat_density: float, n: int, random_state: np.random.RandomState
) -> scipy.sparse.spmatrix:
    """Random sparse symmetric positive definite matrix of dimension :func:`n`, sampled
    from :func:`random_state`."""
    return random_sparse_spd_matrix(
        dim=n, random_state=random_state, density=sparsemat_density
    )


@pytest.fixture(
    params=[
        pytest.param(kernel, id=str(kernel))
        for kernel in [
            kernels.Linear,
            kernels.ExpQuad,
            kernels.RatQuad,
            kernels.Matern,
            kernels.Polynomial,
        ]
    ]
)
def kernel_mat(kernel, n: int, random_state: np.random.RandomState) -> np.ndarray:
    """Kernel matrix evaluated on a randomly drawn data set."""
    data = random_state.uniform(-1.0, 1.0, (n, 1))
    kernel_mat = kernel(input_dim=1)(data)
    return kernel_mat


##################
# Linear systems #
##################


@pytest.fixture()
def linsys_poisson() -> LinearSystem:
    """Discretized Poisson equation with Dirichlet boundary conditions."""
    fpath = os.path.join(os.path.dirname(__file__), "../resources")
    A = scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")
    b = np.load(file=fpath + "/rhs_poisson.npy")
    return LinearSystem(
        A=A,
        solution=scipy.sparse.linalg.spsolve(A=A, b=b),
        b=b,
    )


@pytest.fixture()
def linsys_spd(spd_mat, random_state: np.random.RandomState) -> LinearSystem:
    """Symmetric positive definite linear system."""
    return LinearSystem.from_matrix(A=spd_mat, random_state=random_state)


@pytest.fixture()
def linsys_sparse_spd(
    sparse_spd_mat, random_state: np.random.RandomState
) -> LinearSystem:
    """Sparse symmetric positive definite linear system."""
    return LinearSystem.from_matrix(A=sparse_spd_mat, random_state=random_state)


@pytest.fixture()
def linsys_kernel(kernel_mat, random_state: np.random.RandomState) -> LinearSystem:
    """Linear system with a kernel matrix."""
    return LinearSystem.from_matrix(A=kernel_mat, random_state=random_state)


###################################################
# Probabilistic linear solvers #
###################################################


@pytest.fixture(
    params=[
        pytest.param(linsolve, id=linsolve.__name__)
        for linsolve in [probnum.linalg.problinsolve]  # , bayescg],
    ]
)
def linsolve(request) -> Callable:
    """Interface functions of probabilistic linear solvers."""
    return request.param
