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


#
# def test_sqrt_kalman():
#     dynmod, measmod, initrv, info = car_tracking()
#     delta_t = info["dt"]
#     tms = np.arange(0, 20, delta_t)
#     states, obs = pnfs.statespace.generate(dynmod, measmod, initrv, tms)
#     sqrt_kal = pnfs.SquareRootKalman(dynmod, measmod, initrv)
#     kal = pnfs.Kalman(dynmod, measmod, initrv)
#
#     sol = sqrt_kal.filtsmooth(obs, tms)
#     sol_old = kal.filtsmooth(obs, tms)
#     H = np.eye(2, 4)
#     pos = sol.state_rvs.mean @ H.T
#     pos_old = sol_old.state_rvs.mean @ H.T
#     #
#     # print(obs)
#     # print(pos - pos_old)
#     # plt.plot(pos[:, 0], pos[:, 1])
#     # plt.show()
#     assert 1 == 0
