"""Fixtures to be shared across all modules in this directory.

Mostly some random variables of matching dimensions.
"""

import numpy as np
import pytest

import probnum.random_variables as pnrv
from probnum.problems.zoo.linalg import random_spd_matrix


@pytest.fixture
def test_ndim():
    """Test dimension."""
    return 4


# A few covariance matrices


@pytest.fixture
def spdmat1(test_ndim):
    return random_spd_matrix(test_ndim)


@pytest.fixture
def spdmat2(test_ndim):
    return random_spd_matrix(test_ndim)


@pytest.fixture
def spdmat3(test_ndim):
    return random_spd_matrix(test_ndim)


@pytest.fixture
def spdmat4(test_ndim):
    return random_spd_matrix(test_ndim)


# A few 'Normal' random variables


@pytest.fixture
def some_normal_rv1(test_ndim, spdmat1):

    return pnrv.Normal(mean=np.random.rand(test_ndim), cov=spdmat1)


@pytest.fixture
def some_normal_rv2(test_ndim, spdmat2):
    return pnrv.Normal(mean=np.random.rand(test_ndim), cov=spdmat2)


@pytest.fixture
def some_normal_rv3(test_ndim, spdmat3):
    return pnrv.Normal(mean=np.random.rand(test_ndim), cov=spdmat3)


@pytest.fixture
def some_normal_rv4(test_ndim, spdmat4):
    return pnrv.Normal(mean=np.random.rand(test_ndim), cov=spdmat4)
