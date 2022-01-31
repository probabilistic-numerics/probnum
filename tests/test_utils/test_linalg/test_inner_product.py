"""Tests for general inner products."""

import numpy as np
import pytest

from probnum.utils.linalg import induced_norm, inner_product


@pytest.fixture(scope="module", params=[1, 10, 50])
def n(request) -> int:
    """Vector size."""
    return request.param


@pytest.fixture(scope="module")
def vector0(n: int) -> np.ndarray:
    rng = np.random.default_rng(86 + n)
    return rng.standard_normal(size=(n,))


@pytest.fixture(scope="module")
def vector1(n: int) -> np.ndarray:
    rng = np.random.default_rng(567 + n)
    return rng.standard_normal(size=(n,))


def test_euclidean_inprod(vector0: np.ndarray, vector1: np.ndarray):
    assert vector0 @ vector1 == pytest.approx(inner_product(v=vector0, w=vector1))


def test_euclidean_norm(vector0: np.ndarray):
    assert np.linalg.norm(vector0, ord=2) == pytest.approx(induced_norm(v=vector0))
