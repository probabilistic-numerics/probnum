"""Test fixtures for the linear algebra test problem zoo."""

import numpy as np
import pytest
import pytest_cases
import scipy.sparse

from probnum.problems.zoo.linalg import (
    SuiteSparseMatrix,
    random_sparse_spd_matrix,
    random_spd_matrix,
    suitesparse_matrix,
)


@pytest_cases.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest_cases.fixture()
@pytest.mark.parametrize("m", [1, 2, 25, 100, 250])
def m_rows(m: int) -> int:
    return m


@pytest_cases.fixture()
@pytest.mark.parametrize("n", [1, 2, 25, 100, 250])
def n_cols(n: int) -> int:
    return n


@pytest_cases.fixture()
@pytest.mark.parametrize("density", [0.1, 0.01])
def density(density: float) -> float:
    """Density of a sparse matrix."""
    return density


@pytest_cases.fixture()
def rnd_dense_spd_mat(n_cols: int, rng: np.random.Generator) -> np.ndarray:
    """Random spd matrix generated from :meth:`random_spd_matrix`."""
    return random_spd_matrix(rng=rng, dim=n_cols)


@pytest_cases.fixture()
def rnd_sparse_spd_mat(
    n_cols: int, density: float, rng: np.random.Generator
) -> scipy.sparse.spmatrix:
    """Random sparse spd matrix generated from :meth:`random_sparse_spd_matrix`."""
    return random_sparse_spd_matrix(rng=rng, dim=n_cols, density=density)


rnd_spd_mat = pytest_cases.fixture_union(
    "spd_mat", [rnd_dense_spd_mat, rnd_sparse_spd_mat]
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
