import numpy as np
import pytest

from probnum import filtsmooth

from ..filtsmooth_testcases import car_tracking, ornstein_uhlenbeck

# Problems


@pytest.fixture(params=[car_tracking, ornstein_uhlenbeck])
def setup(request):
    """Filter and regression problem."""
    problem = request.param
    dynmod, measmod, initrv, regression_problem = problem()

    kalman = filtsmooth.Kalman(dynmod, measmod, initrv)
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
