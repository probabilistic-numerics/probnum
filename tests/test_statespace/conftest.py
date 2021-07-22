"""Fixtures to be shared across all modules in this directory.

Mostly some random variables of matching dimensions.
"""

import numpy as np
import pytest

from probnum import randvars
from probnum.problems.zoo.linalg import random_spd_matrix


@pytest.fixture
def rng():
    return np.random.default_rng(seed=123)


@pytest.fixture(params=[2])
def test_ndim(request):
    """Test dimension."""
    return request.param


# A few covariance matrices


@pytest.fixture
def spdmat1(test_ndim, rng):
    return random_spd_matrix(rng, dim=test_ndim)


@pytest.fixture
def spdmat2(test_ndim, rng):
    return random_spd_matrix(rng, dim=test_ndim)


@pytest.fixture
def spdmat3(test_ndim, rng):
    return random_spd_matrix(rng, dim=test_ndim)


@pytest.fixture
def spdmat4(test_ndim, rng):
    return random_spd_matrix(rng, dim=test_ndim)


# A few 'Normal' random variables


@pytest.fixture
def some_normal_rv1(test_ndim, spdmat1, rng):

    return randvars.Normal(
        mean=rng.uniform(size=test_ndim),
        cov=spdmat1,
        cov_cholesky=np.linalg.cholesky(spdmat1),
    )


@pytest.fixture
def some_normal_rv2(test_ndim, spdmat2, rng):
    return randvars.Normal(
        mean=rng.uniform(size=test_ndim),
        cov=spdmat2,
        cov_cholesky=np.linalg.cholesky(spdmat2),
    )


@pytest.fixture
def some_normal_rv3(test_ndim, spdmat3, rng):
    return randvars.Normal(
        mean=rng.uniform(size=test_ndim),
        cov=spdmat3,
        cov_cholesky=np.linalg.cholesky(spdmat3),
    )


@pytest.fixture
def some_normal_rv4(test_ndim, spdmat4, rng):
    return randvars.Normal(
        mean=rng.uniform(size=test_ndim),
        cov=spdmat4,
        cov_cholesky=np.linalg.cholesky(spdmat4),
    )


@pytest.fixture
def diffusion():
    """A diffusion != 1 makes it easier to see if _diffusion is actually used in forward
    and backward."""
    return 5.1412512431


@pytest.fixture(params=["classic", "sqrt"])
def forw_impl_string_linear_gauss(request):
    """Forward implementation choices passed via strings."""
    return request.param


@pytest.fixture(params=["classic", "joseph", "sqrt"])
def backw_impl_string_linear_gauss(request):
    """Backward implementation choices passed via strings."""
    return request.param
