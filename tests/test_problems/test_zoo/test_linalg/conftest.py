"""Test fixtures for the linear algebra test problem zoo."""

import pytest

from probnum.problems.zoo.linalg import SuiteSparseMatrix, suitesparse_matrix


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
