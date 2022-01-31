"""Tests for general inner products."""

import numpy as np
import pytest

from probnum.utils.linalg import induced_norm, inner_product


@pytest.fixture(scope="module", params=[1, 10, 50])
def n(request) -> int:
    """Vector size."""
    return request.param


@pytest.fixture(scope="module", params=[1, 3, 5])
def m(request) -> int:
    """Number of simultaneous vectors."""
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def p(request) -> int:
    """Number of matrices."""
    return request.param


@pytest.fixture(scope="module")
def vector0(n: int) -> np.ndarray:
    rng = np.random.default_rng(86 + n)
    return rng.standard_normal(size=(n,))


@pytest.fixture(scope="module")
def vector1(n: int) -> np.ndarray:
    rng = np.random.default_rng(567 + n)
    return rng.standard_normal(size=(n,))


@pytest.fixture(scope="module")
def array0(p: int, m: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(86 + p + m + n)
    return rng.standard_normal(size=(p, m, n))


@pytest.fixture(scope="module")
def array1(m: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(567 + m + n)
    return rng.standard_normal(size=(m, n))


def test_inner_product_vectors(vector0: np.ndarray, vector1: np.ndarray):
    assert inner_product(v=vector0, w=vector1) == pytest.approx(
        np.inner(vector0, vector1)
    )


def test_inner_product_arrays(array0: np.ndarray, array1: np.ndarray):
    assert inner_product(v=array0, w=array1) == pytest.approx(np.inner(array0, array1))


def test_induced_norm_vector(vector0: np.ndarray):
    assert np.linalg.norm(vector0, ord=2) == pytest.approx(induced_norm(v=vector0))


@pytest.mark.parametrize("axis", [-1, 0])
def test_induced_norm_array(array0: np.ndarray, axis: int):
    assert np.linalg.norm(array0, axis=axis, ord=2) == pytest.approx(
        induced_norm(v=array0, axis=axis)
    )
