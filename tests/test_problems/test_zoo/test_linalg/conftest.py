"""Test fixtures for the linear algebra problem zoo."""

import numpy as np
import pytest
import scipy.sparse

from probnum.problems.zoo.linalg import (
    SuiteSparseMatrix,
    random_sparse_spd_matrix,
    random_spd_matrix,
    suitesparse_matrix,
)


@pytest.fixture(params=[pytest.param(n, id=f"dim{n}") for n in [2, 5, 10, 50, 100]])
def n(request) -> int:
    """Number of columns of the matrix.

    This is mostly used for test parameterization.
    """
    return request.param


@pytest.fixture(params=[pytest.param(seed, id=f"seed{seed}") for seed in range(3)])
def random_state(request) -> np.random.RandomState:
    """Random states used to sample the test case input matrices.

    This is mostly used for test parameterization.
    """
    return np.random.RandomState(seed=request.param)


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
        pytest.param(namegroup, id=f"{namegroup[0]}") for namegroup in (("wm1", "HB"),)
    ],
    name="suitesparse_mat",
)
def fixture_suitesparse_mat(request) -> SuiteSparseMatrix:
    return suitesparse_matrix(name=request.param[0], group=request.param[1])


@pytest.fixture(name="suitesparse_mycielskian")
def fixture_suitesparse_mycielskian() -> SuiteSparseMatrix:
    return suitesparse_matrix(group="Mycielski", name="mycielskian3", verbose=True)
