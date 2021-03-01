import numpy as np
import pytest

import probnum.utils.linalg as utlin
from probnum.problems.zoo.linalg import random_spd_matrix


@pytest.fixture
def even_ndim():
    """Even dimension for the tests, because it is halfed in test_cholesky_optional
    below."""
    return 10


@pytest.fixture
def spdmat1(even_ndim):
    return random_spd_matrix(even_ndim)


@pytest.fixture
def spdmat2(even_ndim):
    return random_spd_matrix(even_ndim)


def test_cholesky_update(spdmat1, spdmat2):
    expected = np.linalg.cholesky(spdmat1 + spdmat2)

    S1 = np.linalg.cholesky(spdmat1)
    S2 = np.linalg.cholesky(spdmat2)
    received = utlin.cholesky_update(S1, S2)
    np.testing.assert_allclose(expected, received)


def test_cholesky_optional(spdmat1, even_ndim):
    """Assert that cholesky_update() transforms a non-square matrix square-root into a
    correct Cholesky factor."""
    H = np.random.rand(int(0.5 * even_ndim), even_ndim)
    expected = np.linalg.cholesky(H @ spdmat1 @ H.T)
    S1 = np.linalg.cholesky(spdmat1)
    received = utlin.cholesky_update(H @ S1)
    np.testing.assert_allclose(expected, received)


def test_triu_to_tril():

    # Make a random tril matrix
    mat = np.tril(np.random.rand(4, 4))
    scale = np.array([1.0, 1.0, 1e-5, 1e-5])
    tril = mat @ np.diag(scale)

    # Turn it into a triu matrix with non-positive diagonals
    signs = np.array([1.0, -1.0, -1.0, -1.0])
    triu = (tril @ np.diag(signs)).T

    # Call triu_to_positive_til
    tril_received = utlin.triu_to_positive_tril(triu)

    # Sanity checks
    np.testing.assert_allclose(triu.T @ triu, tril @ tril.T)
    np.testing.assert_allclose(triu.T @ triu, tril_received @ tril_received.T)

    # Assert that the initial tril matrix comes out
    np.testing.assert_allclose(tril_received, tril)
