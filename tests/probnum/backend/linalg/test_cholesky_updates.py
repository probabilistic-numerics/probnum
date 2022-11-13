from probnum import backend, compat
from probnum.problems.zoo.linalg import random_spd_matrix

import pytest
import tests.utils


@pytest.fixture
def even_ndim():
    """Even dimension for the tests, because it is halfed in test_cholesky_optional
    below."""
    return 10


@pytest.fixture
def spdmats(even_ndim):
    rng_state = tests.utils.random.rng_state_from_sampling_args(
        base_seed=3897, shape=even_ndim
    )
    rng_state1, rng_state2 = backend.random.split(rng_state, num=2)

    spdmat1 = random_spd_matrix(rng_state1, shape=(even_ndim, even_ndim))
    spdmat2 = random_spd_matrix(rng_state2, shape=(even_ndim, even_ndim))

    return spdmat1, spdmat2


@pytest.fixture
def spdmat1(spdmats):
    return spdmats[0]


@pytest.fixture
def spdmat2(spdmats):
    return spdmats[1]


def test_cholesky_update(spdmat1, spdmat2):
    expected = backend.linalg.cholesky(spdmat1 + spdmat2, upper=False)

    S1 = backend.linalg.cholesky(spdmat1, upper=False)
    S2 = backend.linalg.cholesky(spdmat2, upper=False)
    received = backend.linalg.cholesky_update(S1, S2)
    compat.testing.assert_allclose(expected, received)


def test_cholesky_optional(spdmat1, even_ndim):
    """Assert that cholesky_update() transforms a non-square matrix square-root into a
    correct Cholesky factor."""
    H_shape = (even_ndim // 2, even_ndim)
    H = backend.random.uniform(
        rng_state=tests.utils.random.rng_state_from_sampling_args(
            base_seed=2908,
            shape=H_shape,
        ),
        shape=H_shape,
    )
    expected = backend.linalg.cholesky(H @ spdmat1 @ H.T, upper=False)
    S1 = backend.linalg.cholesky(spdmat1, upper=False)
    received = backend.linalg.cholesky_update(H @ S1)
    compat.testing.assert_allclose(expected, received)


def test_tril_to_positive_tril():

    # Make a random tril matrix
    mat = backend.tril(
        backend.random.uniform(rng_state=backend.random.rng_state(4897), shape=(4, 4))
    )
    scale = backend.asarray([1.0, 1.0, 1e-5, 1e-5])
    signs = backend.asarray([1.0, -1.0, -1.0, -1.0])
    tril = mat @ backend.diag(scale)
    tril_wrong_signs = tril @ backend.diag(signs)

    # Call triu_to_positive_til
    tril_received = backend.linalg.tril_to_positive_tril(tril_wrong_signs)

    # Sanity check
    compat.testing.assert_allclose(tril @ tril.T, tril_received @ tril_received.T)

    # Assert that the initial tril matrix comes out
    compat.testing.assert_allclose(tril_received, tril)