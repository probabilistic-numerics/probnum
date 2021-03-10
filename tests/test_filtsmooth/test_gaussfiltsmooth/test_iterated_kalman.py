import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss

from .filtsmooth_testcases import logistic_ode, pendulum


def pendulum_problem():
    """Pendulum problem."""
    problem = pendulum()
    dynmod, measmod, initrv, info = problem
    dynmod = pnfs.DiscreteEKFComponent(dynmod)
    measmod = pnfs.DiscreteEKFComponent(measmod)

    times = np.arange(0, info["tmax"], info["dt"])
    states, obs = pnss.generate_samples(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    return dynmod, measmod, initrv, info, obs, times, states


def logistic_ode_problem():
    """Logistic ODE problem."""
    problem = logistic_ode()
    dynmod, measmod, initrv, info = problem

    times = np.arange(0, info["tmax"], info["dt"])
    obs = np.zeros((len(times), 1))

    states = info["ode"].solution(times)
    return dynmod, measmod, initrv, info, obs, times, states


@pytest.fixture
def kalman(problem):
    """Create a Kalman object."""
    dynmod, measmod, initrv, info, *_ = problem
    return pnfs.Kalman(dynmod, measmod, initrv)


@pytest.mark.parametrize("problem", [logistic_ode_problem(), pendulum_problem()])
def test_rmse_filt_smooth(kalman, problem):
    """Assert that iterated smoothing beats smoothing beats filtering."""
    *_, obs, times, truth = problem

    stopcrit = pnfs.StoppingCriterion(atol=1e-1, rtol=1e-1, maxit=10)

    posterior = kalman.filter(obs, times)
    posterior = kalman.smooth(posterior)

    iterated_posterior = kalman.iterated_filtsmooth(obs, times, stopcrit=stopcrit)

    filtms = posterior.filtering_posterior.state_rvs.mean
    smooms = posterior.state_rvs.mean
    iterms = iterated_posterior.state_rvs.mean

    if filtms.ndim == 1:
        filtms = filtms.reshape((-1, 1))
        smooms = smooms.reshape((-1, 1))
        iterms = iterms.reshape((-1, 1))

    if truth.ndim == 1:
        truth = truth.reshape((-1, 1))

    # Compare only zeroth component
    # for compatibility with all test cases
    filtms_rmse = np.mean(np.abs(filtms[:, 0] - truth[:, 0]))
    smooms_rmse = np.mean(np.abs(smooms[:, 0] - truth[:, 0]))
    iterms_rmse = np.mean(np.abs(iterms[:, 0] - truth[:, 0]))

    assert iterms_rmse < smooms_rmse < filtms_rmse
