import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.filtsmooth.statespace as pnfss

from .filtsmooth_testcases import car_tracking, ornstein_uhlenbeck


@pytest.fixture
def problem():
    return car_tracking()


@pytest.fixture
def problem():
    return ornstein_uhlenbeck()


@pytest.fixture
def kalman(problem):
    dynmod, measmod, initrv, info = problem
    return pnfs.Kalman(dynmod, measmod, initrv)


@pytest.fixture
def data(problem):
    dynmod, measmod, initrv, info = problem
    times = np.arange(0, info["tmax"], info["dt"])
    states, obs = pnfss.generate(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    return obs, times, states


def test_rmse_filt_smooth(kalman, data):
    """Check that smoothing RMSE is better than filtering RMSE is better than
    observation RMSE."""
    obs, times, truth = data

    filter_posterior = kalman.filter(obs, times)
    smooth_posterior = kalman.smooth(filter_posterior)

    filtms = filter_posterior.state_rvs.mean
    smooms = smooth_posterior.state_rvs.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(obs - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse
