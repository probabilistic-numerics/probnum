"""Test fixtures for the linalg subpackage."""

import os
from typing import Callable, Union

import numpy as np
import pytest
import scipy.sparse

import probnum
import probnum.kernels as kernels
import probnum.linops as linops
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix
from probnum.type import RandomStateArgType


@pytest.fixture(
    params=[pytest.param(n, id=f"dim{n}") for n in [5, 10, 50, 100]], name="n"
)
def fixture_n(request) -> int:
    """Number of columns of the system matrix.

    This is mostly used for test parameterization.
    """
    return request.param


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in range(3)],
    name="random_state",
)
def fixture_random_state(request) -> np.random.RandomState:
    """Random states used to sample the test case input matrices.

    This is mostly used for test parameterization.
    """
    return np.random.RandomState(seed=request.param)


############
# Matrices #
############
def random_data(n: int, p: int, random_state: RandomStateArgType = None):
    """Generate a random n x p data matrix."""
    return probnum.utils.as_random_state(random_state).uniform(-1.0, 1.0, (n, p))


@pytest.fixture(
    params=[
        pytest.param(mat, id=mat[0])
        for mat in [
            ("spd", random_spd_matrix(dim=100, random_state=1)),
            (
                "sparsespd",
                random_sparse_spd_matrix(dim=1000, density=0.01, random_state=1),
            ),
            (
                "expquad",
                kernels.ExpQuad(input_dim=2)(random_data(n=10, p=2, random_state=1)),
            ),
            (
                "ratquad",
                kernels.RatQuad(input_dim=3)(random_data(n=50, p=3, random_state=1)),
            ),
            (
                "matern",
                kernels.Matern(input_dim=2)(random_data(n=100, p=2, random_state=1)),
            ),
            ("idop", linops.Identity(20)),
            ("matop", linops.MatrixMult(random_spd_matrix(10, random_state=1))),
        ]
    ],
    name="mat",
)
def fixture_mat(
    request,
) -> Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator]:
    """(Sparse) matrix or linear operator."""
    return request.param[1]


@pytest.fixture(name="spd_mat")
def fixture_spd_mat(n: int, random_state: np.random.RandomState) -> np.ndarray:
    """Random symmetric positive definite matrix of dimension :func:`n`, sampled from
    :func:`random_state`."""
    return random_spd_matrix(dim=n, random_state=random_state)


@pytest.fixture(
    params=[
        pytest.param(sparsemat_density, id=f"density{sparsemat_density}")
        for sparsemat_density in (0.001, 0.01, 0.1)
    ],
    name="sparsemat_density",
)
def fixture_sparsemat_density(request):
    """Density of a sparse matrix defined by the fraction of nonzero entries."""
    return request.param


@pytest.fixture(name="sparse_spd_mat")
def fixture_sparse_spd_mat(
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
    ],
    name="kernel_mat",
)
def fixture_kernel_mat(
    request, n: int, random_state: np.random.RandomState
) -> np.ndarray:
    """Kernel matrix evaluated on a randomly drawn data set."""
    data = random_data(n=n, p=1, random_state=random_state)
    kernel_mat = request.param(input_dim=1)(data)
    return kernel_mat


##################
# Linear systems #
##################


@pytest.fixture(name="linsys")
def fixture_linsys(
    mat: Union[np.ndarray, scipy.sparse.spmatrix, linops.LinearOperator],
    random_state: np.random.RandomState,
) -> LinearSystem:
    """Random linear system."""
    return LinearSystem.from_matrix(A=mat, random_state=random_state)


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
