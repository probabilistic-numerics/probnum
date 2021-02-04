import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.filtsmooth.statespace as pnfss

from .filtsmooth_testcases import car_tracking, ornstein_uhlenbeck


@pytest.fixture
def problem():
    """Car tracking problem."""
    return car_tracking()


@pytest.fixture
def problem():
    """Ornstein-Uhlenbeck problem."""
    return ornstein_uhlenbeck()


@pytest.fixture
def update():
    """The usual Kalman update.

    Yields Kalman filter.
    """
    return pnfs.update_classic


@pytest.fixture
def update():
    """The usual Kalman update.

    Yields Kalman filter.
    """
    return pnfs.update_sqrt


@pytest.fixture
def predict():
    return pnfs.predict_via_transition


@pytest.fixture
def predict():
    return pnfs.predict_sqrt


@pytest.fixture
def measure():
    return pnfs.measure_via_transition


@pytest.fixture
def measure():
    return pnfs.measure_sqrt


@pytest.fixture
def update():
    """The usual Kalman update.

    Yields Kalman filter.
    """
    return pnfs.update_sqrt


@pytest.fixture
def update():
    """Iterated classical update.

    Yields I(E/U)KF depending on the approximate measurement model.
    """
    stopcrit = pnfs.StoppingCriterion()
    return pnfs.iterate_update(pnfs.update_classic, stopcrit=stopcrit)


@pytest.fixture
def smooth_step():
    return pnfs.rts_smooth_step_classic


@pytest.fixture
def smooth_step():
    return pnfs.rts_smooth_step_sqrt


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


def test_rmse_filt_smooth(kalman, data):
    """Assert that smoothing beats filtering beats nothing."""
    obs, times, truth = data

    filter_posterior = kalman.filter(obs, times)
    smooth_posterior = kalman.smooth(filter_posterior)

    filtms = filter_posterior.state_rvs.mean
    smooms = smooth_posterior.state_rvs.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(obs - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse
