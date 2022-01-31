"""Tests for general inner products."""

import numpy as np
import pytest

from probnum.problems.zoo.linalg import random_spd_matrix
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


def test_euclidean_norm_vector(vector0: np.ndarray):
    assert np.linalg.norm(vector0, ord=2) == pytest.approx(induced_norm(v=vector0))


@pytest.mark.parametrize("axis", [0, 1])
def test_euclidean_norm_array(array0: np.ndarray, axis: int):
    assert np.linalg.norm(array0, axis=axis, ord=2) == pytest.approx(
        induced_norm(v=array0, axis=axis)
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_induced_norm_array(array0: np.ndarray, axis: int):
    inprod_mat = random_spd_matrix(
        rng=np.random.default_rng(254), dim=array0.shape[axis]
    )
    array0_moved_axis = np.moveaxis(array0, axis, -1)
    A_array_0_moved_axis = np.squeeze(
        inprod_mat @ array0_moved_axis[..., :, None], axis=-1
    )

    assert np.sqrt(
        np.sum(array0_moved_axis * A_array_0_moved_axis, axis=-1)
    ) == pytest.approx(induced_norm(v=array0, A=inprod_mat, axis=axis))
