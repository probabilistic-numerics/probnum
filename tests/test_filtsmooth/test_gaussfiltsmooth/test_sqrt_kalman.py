"""Tests for square-root Kalman filtering and smoothing.

Check that the output matches the output of standard Kalman filtering
and smoothing.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv
from probnum.problems.zoo.linalg import random_spd_matrix

from .filtsmooth_testcases import car_tracking

np.random.seed(42)


@pytest.fixture
def dynmod():
    return car_tracking()[0]


@pytest.fixture
def measmod():
    return car_tracking()[1]


@pytest.fixture
def initrv():
    return car_tracking()[2]


@pytest.fixture
def info():
    return car_tracking()[3]


@pytest.fixture
def random_rv4():
    covmat = random_spd_matrix(4)
    mean = np.random.rand(4)
    return pnrv.Normal(mean, covmat)


@pytest.fixture
def sqrt_kalman(dynmod, measmod, initrv):
    return pnfs.SquareRootKalman(dynmod, measmod, initrv)


@pytest.fixture
def kalman(dynmod, measmod, initrv):
    return pnfs.Kalman(dynmod, measmod, initrv)


def test_predict(sqrt_kalman, kalman, random_rv4):
    res1, info1 = sqrt_kalman.predict(0.0, 1.0, random_rv4)
    res2, info2 = kalman.predict(0.0, 1.0, random_rv4)

    np.testing.assert_allclose(res1.mean, res2.mean)
    np.testing.assert_allclose(res1.cov, res2.cov)
    np.testing.assert_allclose(res1.cov_cholesky, res2.cov_cholesky)
    np.testing.assert_allclose(info1["crosscov"], info2["crosscov"])


def test_measure(sqrt_kalman, kalman, random_rv4):
    res1, info1 = sqrt_kalman.measure(1.0, random_rv4)
    res2, info2 = kalman.measure(1.0, random_rv4)

    np.testing.assert_allclose(res1.mean, res2.mean)
    np.testing.assert_allclose(res1.cov, res2.cov)
    np.testing.assert_allclose(res1.cov_cholesky, res2.cov_cholesky)
    np.testing.assert_allclose(info1["crosscov"], info2["crosscov"])


@pytest.fixture
def random_data2():
    return np.random.rand(2)


def test_update(sqrt_kalman, kalman, random_rv4, random_data2):
    res1, meas_rv1, _ = sqrt_kalman.update(1.0, random_rv4, random_data2)
    res2, meas_rv2, _ = kalman.update(1.0, random_rv4, random_data2)

    np.testing.assert_allclose(meas_rv1.cov, meas_rv2.cov)
    np.testing.assert_allclose(meas_rv1.cov_cholesky, meas_rv2.cov_cholesky)
    np.testing.assert_allclose(meas_rv1.mean, meas_rv2.mean)

    np.testing.assert_allclose(res1.cov_cholesky, res2.cov_cholesky)
    np.testing.assert_allclose(res1.cov, res2.cov)
    np.testing.assert_allclose(res1.mean, res2.mean)


@pytest.fixture
def times(dynmod, measmod, initrv, info):
    delta_t = info["dt"]
    return np.arange(0, 20, delta_t)


@pytest.fixture
def data(dynmod, measmod, initrv, times):
    states, obs = pnfs.statespace.generate(dynmod, measmod, initrv, times)
    return obs


def test_filter(sqrt_kalman, kalman, data, times):
    sol_sqrt = sqrt_kalman.filter(data, times)
    sol_classic = kalman.filter(data, times)
    np.testing.assert_allclose(sol_sqrt.state_rvs.mean, sol_classic.state_rvs.mean)
    np.testing.assert_allclose(sol_sqrt.state_rvs.cov, sol_classic.state_rvs.cov)


def test_filtsmooth(sqrt_kalman, kalman, data, times):
    sol_sqrt = sqrt_kalman.filtsmooth(data, times)
    sol_classic = kalman.filtsmooth(data, times)

    # non-strict rtol parameter because there are many small values
    np.testing.assert_allclose(
        sol_sqrt.state_rvs.mean, sol_classic.state_rvs.mean, rtol=1e-2, atol=1e-12
    )
    np.testing.assert_allclose(
        sol_sqrt.state_rvs.cov, sol_classic.state_rvs.cov, rtol=1e-2, atol=1e-12
    )
