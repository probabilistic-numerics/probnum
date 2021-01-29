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

from .filtsmooth_testcases import car_tracking, pendulum

np.random.seed(42)


@pytest.fixture
def problem():
    """Car-tracking problem."""
    return car_tracking


@pytest.fixture
def d_dynamics(problem):
    """Dimension of the dynamics model of given problem."""
    return len(problem()[0].dynamicsmat)


@pytest.fixture
def d_measurements(problem):
    """Dimension of the measurement model of given problem."""
    return len(problem()[1].dynamicsmat)


@pytest.fixture
def info(problem):
    return problem()[3]


@pytest.fixture
def random_rv(d_dynamics):
    covmat = random_spd_matrix(d_dynamics)
    mean = np.random.rand(d_dynamics)
    return pnrv.Normal(mean, covmat)


@pytest.fixture
def both_filters(problem):
    dynmod, measmod, initrv, _ = problem()
    sqrt_kalman = pnfs.SquareRootKalman(dynmod, measmod, initrv)
    kalman = pnfs.Kalman(dynmod, measmod, initrv)
    return (sqrt_kalman, kalman)


def test_predict(both_filters, random_rv):
    sqrt_kalman, kalman = both_filters
    res1, info1 = sqrt_kalman.predict(0.0, 1.0, random_rv)
    res2, info2 = kalman.predict(0.0, 1.0, random_rv)

    np.testing.assert_allclose(res1.mean, res2.mean)
    np.testing.assert_allclose(res1.cov, res2.cov)
    np.testing.assert_allclose(res1.cov_cholesky, res2.cov_cholesky)
    np.testing.assert_allclose(info1["crosscov"], info2["crosscov"])


def test_measure(both_filters, random_rv):
    sqrt_kalman, kalman = both_filters

    res1, info1 = sqrt_kalman.measure(1.0, random_rv)
    res2, info2 = kalman.measure(1.0, random_rv)

    np.testing.assert_allclose(res1.mean, res2.mean)
    np.testing.assert_allclose(res1.cov, res2.cov)
    np.testing.assert_allclose(res1.cov_cholesky, res2.cov_cholesky)
    np.testing.assert_allclose(info1["crosscov"], info2["crosscov"])


@pytest.fixture
def random_observations(d_measurements):
    return np.random.rand(d_measurements)


def test_update(both_filters, random_rv, random_observations):
    sqrt_kalman, kalman = both_filters
    res1, meas_rv1, _ = sqrt_kalman.update(1.0, random_rv, random_observations)
    res2, meas_rv2, _ = kalman.update(1.0, random_rv, random_observations)

    np.testing.assert_allclose(meas_rv1.cov, meas_rv2.cov)
    np.testing.assert_allclose(meas_rv1.cov_cholesky, meas_rv2.cov_cholesky)
    np.testing.assert_allclose(meas_rv1.mean, meas_rv2.mean)

    np.testing.assert_allclose(res1.cov_cholesky, res2.cov_cholesky)
    np.testing.assert_allclose(res1.cov, res2.cov)
    np.testing.assert_allclose(res1.mean, res2.mean)


@pytest.fixture
def times_data(problem):
    dynmod, measmod, initrv, info = problem()
    delta_t = info["dt"]

    times = np.arange(0, 20, delta_t)
    states, obs = pnfs.statespace.generate(dynmod, measmod, initrv, times)
    return times, obs


def test_filter(both_filters, times_data):
    times, data = times_data
    sqrt_kalman, kalman = both_filters
    sol_sqrt = sqrt_kalman.filter(data, times)
    sol_classic = kalman.filter(data, times)
    np.testing.assert_allclose(sol_sqrt.state_rvs.mean, sol_classic.state_rvs.mean)
    np.testing.assert_allclose(sol_sqrt.state_rvs.cov, sol_classic.state_rvs.cov)


def test_filtsmooth(both_filters, times_data):
    times, data = times_data
    sqrt_kalman, kalman = both_filters
    sol_sqrt = sqrt_kalman.filtsmooth(data, times)
    sol_classic = kalman.filtsmooth(data, times)

    # non-strict rtol parameter because there are many small values
    np.testing.assert_allclose(
        sol_sqrt.state_rvs.mean, sol_classic.state_rvs.mean, rtol=1e-2, atol=1e-12
    )
    np.testing.assert_allclose(
        sol_sqrt.state_rvs.cov, sol_classic.state_rvs.cov, rtol=1e-2, atol=1e-12
    )
