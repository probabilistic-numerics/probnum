"""Test fixtures for linear solvers."""
from pytest_cases import fixture, parametrize


@fixture()
@parametrize("m", [1, 10, 100])
def nrows(m: int) -> int:
    """Number of rows of a matrix."""
    return m


@fixture()
@parametrize("n", [1, 10, 100])
def ncols(n: int) -> int:
    """Number of columns of a matrix."""
    return n


@fixture
@parametrize("k", [1, 2, 100])
def nrhs(k: int) -> int:
    """Number of right-hand-sides of a linear system."""
    return k
