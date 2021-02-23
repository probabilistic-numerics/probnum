import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss

from .filtsmooth_testcases import car_tracking, ornstein_uhlenbeck

# Problems


@pytest.fixture
def problem():
    """Car tracking problem."""
    return car_tracking()


@pytest.fixture
def problem():
    """Ornstein-Uhlenbeck problem."""
    return ornstein_uhlenbeck()


# Test setup


@pytest.fixture
def kalman(problem):
    """Create a Kalman object."""
    dynmod, measmod, initrv, info = problem
    return pnfs.Kalman(
        dynmod,
        measmod,
        initrv,
    )


@pytest.fixture
def data(problem):
    """Create artificial data."""
    dynmod, measmod, initrv, info = problem
    times = np.arange(0, info["tmax"], info["dt"])
    states, obs = pnss.generate_samples(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    return obs, times, states


# Real test stuff


def test_rmse_filt_smooth(kalman, data):
    """Assert that smoothing beats filtering beats nothing."""
    obs, times, truth = data

    posterior = kalman.filtsmooth(obs, times)

    filtms = posterior.filtering_posterior.state_rvs.mean
    smooms = posterior.state_rvs.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(obs - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse
