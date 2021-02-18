import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.filtsmooth.statespace as pnfss

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


# Predictions


@pytest.fixture
def predict():
    """Classical prediction."""
    return pnfs.predict_via_transition


@pytest.fixture
def predict():
    """Square-root prediction."""
    return pnfs.predict_sqrt


# Measures


@pytest.fixture
def measure():
    """Classical measure."""
    return pnfs.measure_via_transition


@pytest.fixture
def measure():
    """Square-root measure."""
    return pnfs.measure_sqrt


# Updates


@pytest.fixture
def update():
    """Classical update."""
    return pnfs.update_classic


@pytest.fixture
def update():
    """Square-root update."""
    return pnfs.update_sqrt


@pytest.fixture
def update():
    """Iterated classical update."""
    stopcrit = pnfs.StoppingCriterion()
    return pnfs.iterate_update(pnfs.update_classic, stopcrit=stopcrit)


# Smoothing steps


@pytest.fixture
def smooth_step():
    """Classical smoothing step."""
    return pnfs.rts_smooth_step_classic


@pytest.fixture
def smooth_step():
    """Square-root smoothing step."""
    return pnfs.rts_smooth_step_sqrt


# Test setup


@pytest.fixture
def kalman(problem, predict, measure, update, smooth_step):
    """Create a Kalman object."""
    dynmod, measmod, initrv, info = problem
    return pnfs.Kalman(
        dynmod,
        measmod,
        initrv,
        use_predict=predict,
        use_measure=measure,
        use_update=update,
        use_smooth_step=smooth_step,
    )


@pytest.fixture
def data(problem):
    """Create artificial data."""
    dynmod, measmod, initrv, info = problem
    times = np.arange(0, info["tmax"], info["dt"])
    states, obs = pnfss.generate(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    return obs, times, states


# Real test stuff


def test_rmse_filt_smooth(kalman, data):
    """Assert that smoothing beats filtering beats nothing."""
    obs, times, truth = data

    posterior = kalman.filter(obs, times)
    posterior = kalman.smooth(posterior)

    filtms = posterior.filtering_posterior.state_rvs.mean
    smooms = posterior.state_rvs.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(obs - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse
