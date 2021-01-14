"""Tests for square-root Kalman filtering and smoothing.

Check that the output matches the output of standard Kalman filtering
and smoothing.
"""
# pylint: disable=redefined-outer-name

import numpy as np
import pytest  # pylint: disable=import-error

import probnum.filtsmooth as pnfs
import probnum.random_variables as pnrv
from probnum.problems.zoo.linalg import random_spd_matrix

from .filtsmooth_testcases import car_tracking

np.random.seed(42)


@pytest.fixture
def problem():
    return car_tracking


@pytest.fixture
def d_dynamics(problem):
    return len(problem()[0].dynamicsmat)


@pytest.fixture
def d_measurements(problem):
    return len(problem()[1].dynamicsmat)


@pytest.fixture
def dynmod(problem):
    return problem()[0]


@pytest.fixture
def measmod(problem):
    return problem()[1]


@pytest.fixture
def initrv(problem):
    return problem()[2]


@pytest.fixture
def info(problem):
    return problem()[3]


@pytest.fixture
def random_rv(d_dynamics):
    covmat = random_spd_matrix(d_dynamics)
    mean = np.random.rand(d_dynamics)
    return pnrv.Normal(mean, covmat)


@pytest.fixture
def sqrt_kalman(dynmod, measmod, initrv):
    return pnfs.SquareRootKalman(dynmod, measmod, initrv)


@pytest.fixture
def kalman(dynmod, measmod, initrv):
    return pnfs.Kalman(dynmod, measmod, initrv)


def test_predict(sqrt_kalman, kalman, random_rv):
    res1, info1 = sqrt_kalman.predict(0.0, 1.0, random_rv)
    res2, info2 = kalman.predict(0.0, 1.0, random_rv)

    np.testing.assert_allclose(res1.mean, res2.mean)
    np.testing.assert_allclose(res1.cov, res2.cov)
    np.testing.assert_allclose(res1.cov_cholesky, res2.cov_cholesky)
    np.testing.assert_allclose(info1["crosscov"], info2["crosscov"])


def test_measure(sqrt_kalman, kalman, random_rv):
    res1, info1 = sqrt_kalman.measure(1.0, random_rv)
    res2, info2 = kalman.measure(1.0, random_rv)

    np.testing.assert_allclose(res1.mean, res2.mean)
    np.testing.assert_allclose(res1.cov, res2.cov)
    np.testing.assert_allclose(res1.cov_cholesky, res2.cov_cholesky)
    np.testing.assert_allclose(info1["crosscov"], info2["crosscov"])


@pytest.fixture
def random_observations(d_measurements):
    return np.random.rand(d_measurements)


def test_update(sqrt_kalman, kalman, random_rv, random_observations):
    res1, meas_rv1, _ = sqrt_kalman.update(1.0, random_rv, random_observations)
    res2, meas_rv2, _ = kalman.update(1.0, random_rv, random_observations)

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
