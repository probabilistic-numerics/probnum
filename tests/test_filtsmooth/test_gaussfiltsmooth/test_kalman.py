import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss
from probnum.problems import RegressionProblem

from ..filtsmooth_testcases import car_tracking, ornstein_uhlenbeck

# Problems


@pytest.fixture
def problem():
    """Car tracking problem."""
    return car_tracking()


@pytest.fixture
def problem():
    """Ornstein-Uhlenbeck problem."""
    return ornstein_uhlenbeck()


@pytest.fixture
def setup(problem):
    """Filter and regression problem."""
    dynmod, measmod, initrv, regression_problem = problem

    kalman = pnfs.Kalman(dynmod, measmod, initrv)
    return kalman, regression_problem


def test_rmse_filt_smooth(setup):
    """Assert that smoothing beats filtering beats nothing."""
    kalman, regression_problem = setup
    truth = regression_problem.solution

    posterior = kalman.filtsmooth(regression_problem)

    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(regression_problem.observations - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse
