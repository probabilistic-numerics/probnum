"""Test fixtures for the linalg subpackage."""

import os

import numpy as np
import pytest
import scipy.sparse

import probnum.kernels as kernels
from probnum.problems import LinearSystem
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix


@pytest.fixture(params=[pytest.param(n, id=f"dim{n}") for n in [5, 10, 50, 100]])
def n(dim) -> int:
    """Number of columns of the system matrix.

    This is mostly used for test parameterization.
    """
    return dim.param


@pytest.fixture(params=[pytest.param(seed, id=f"seed{seed}") for seed in range(5)])
def random_state(seed) -> np.random.RandomState:
    """Random states used to sample the test case input matrices.

    This is mostly used for test parameterization.
    """
    return np.random.RandomState(seed=seed.param)


# Matrices


@pytest.fixture()
def A_spd(n: int, random_state: np.random.RandomState):
    """Random symmetric positive definite matrix of dimension :func:`n`, sampled from
    :func:`random_state`."""
    return random_spd_matrix(dim=n, random_state=random_state)


@pytest.fixture(
    params=[
        pytest.param(density, id=f"density{density}") for density in (0.001, 0.01, 0.1)
    ]
)
def A_sparse_spd(density, n: int, random_state: np.random.RandomState):
    """Random sparse symmetric positive definite matrix of dimension :func:`n`, sampled
    from :func:`random_state`."""
    return random_sparse_spd_matrix(
        dim=n, random_state=random_state, density=density.param
    )


# @pytest.fixture(
#     params=[
#         pytest.param(kernel, id=str(kernel))
#         for kernel in [
#             kernels.Linear,
#             kernels.ExpQuad,
#             kernels.RatQuad,
#             kernels.Matern,
#             kernels.Polynomial,
#         ]
#     ]
# )
# def kernel_mat(kernel, n: int, random_state: np.random.RandomState):
#     """Kernel matrix evaluated on a randomly drawn data set."""
#     data = np.random.uniform(-1.0, 1.0, (n, 1))


# Linear systems


@pytest.fixture()
def linsys_poisson():
    """Discretized Poisson equation with Dirichlet boundary conditions."""
    fpath = os.path.join(os.path.dirname(__file__), "../resources")
    A = scipy.sparse.load_npz(file=fpath + "/matrix_poisson.npz")
    b = np.load(file=fpath + "/rhs_poisson.npy")
    return LinearSystem(
        A=A,
        solution=scipy.sparse.linalg.spsolve(A=A, b=b),
        b=b,
    )
