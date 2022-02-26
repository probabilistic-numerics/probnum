import numpy as np
import pytest

from probnum import backend
from probnum.problems.zoo.linalg import random_spd_matrix
import probnum.utils.linalg as utlin


@pytest.fixture
def even_ndim():
    """Even dimension for the tests, because it is halfed in test_cholesky_optional
    below."""
    return 10


@pytest.fixture
def spdmats(even_ndim):
    seed = backend.random.seed(abs(hash(even_ndim)))
    seed1, seed2 = backend.random.split(seed, num=2)

    spdmat1 = random_spd_matrix(seed1, dim=even_ndim)
    spdmat2 = random_spd_matrix(seed2, dim=even_ndim)

    return spdmat1, spdmat2


@pytest.fixture
def spdmat1(spdmats):
    return spdmats[0]


@pytest.fixture
def spdmat2(spdmats):
    return spdmats[1]


@pytest.mark.skipif_backend(backend.Backend.JAX)
@pytest.mark.skipif_backend(backend.Backend.TORCH)
def test_cholesky_update(spdmat1, spdmat2):
    expected = np.linalg.cholesky(spdmat1 + spdmat2)

    S1 = np.linalg.cholesky(spdmat1)
    S2 = np.linalg.cholesky(spdmat2)
    received = utlin.cholesky_update(S1, S2)
    np.testing.assert_allclose(expected, received)


@pytest.mark.skipif_backend(backend.Backend.JAX)
@pytest.mark.skipif_backend(backend.Backend.TORCH)
def test_cholesky_optional(spdmat1, even_ndim):
    """Assert that cholesky_update() transforms a non-square matrix square-root into a
    correct Cholesky factor."""
    H = np.random.rand(even_ndim // 2, even_ndim)
    expected = np.linalg.cholesky(H @ spdmat1 @ H.T)
    S1 = np.linalg.cholesky(spdmat1)
    received = utlin.cholesky_update(H @ S1)
    np.testing.assert_allclose(expected, received)


@pytest.mark.skipif_backend(backend.Backend.JAX)
@pytest.mark.skipif_backend(backend.Backend.TORCH)
def test_tril_to_positive_tril():

    # Make a random tril matrix
    mat = np.tril(np.random.rand(4, 4))
    scale = np.array([1.0, 1.0, 1e-5, 1e-5])
    signs = np.array([1.0, -1.0, -1.0, -1.0])
    tril = mat @ np.diag(scale)
    tril_wrong_signs = tril @ np.diag(signs)

    # Call triu_to_positive_til
    tril_received = utlin.tril_to_positive_tril(tril_wrong_signs)

    # Sanity check
    np.testing.assert_allclose(tril @ tril.T, tril_received @ tril_received.T)

    # Assert that the initial tril matrix comes out
    np.testing.assert_allclose(tril_received, tril)
