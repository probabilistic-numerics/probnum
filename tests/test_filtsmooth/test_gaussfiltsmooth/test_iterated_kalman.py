import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.filtsmooth.statespace as pnfss

from .filtsmooth_testcases import pendulum


@pytest.fixture
def problem():
    """Car tracking problem."""
    return pendulum()


@pytest.fixture
def update():
    """The usual Kalman update.

    Yields Kalman filter.
    """
    return pnfs.update_classic


@pytest.fixture
def update():
    """Iterated classical update.

    Yields I(E/U)KF depending on the approximate measurement model.
    """
    stopcrit = pnfs.StoppingCriterion()
    return pnfs.iterate_update(pnfs.update_classic, stopcrit=stopcrit)


@pytest.fixture
def kalman(problem, update):
    """Create a Kalman object."""
    dynmod, measmod, initrv, info = problem
    dynmod = pnfs.DiscreteEKFComponent(dynmod)
    measmod = pnfs.DiscreteEKFComponent(measmod)
    kalman = pnfs.Kalman(dynmod, measmod, initrv)
    return pnfs.IteratedKalman(kalman)


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
    iterated_posterior = kalman.iterated_filtsmooth(obs, times)

    filtms = filter_posterior.state_rvs.mean
    smooms = smooth_posterior.state_rvs.mean
    iterms = iterated_posterior.state_rvs.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    iterms_rmse = np.mean(np.abs(iterms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(obs - truth[:, :2]))

    assert iterms_rmse < smooms_rmse < filtms_rmse < obs_rmse
