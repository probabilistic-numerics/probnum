import numpy as np
import pytest

import probnum.filtsmooth as pnfs
from probnum.problems import RegressionProblem

from ..filtsmooth_testcases import logistic_ode


def logistic_ode_problem():
    """Logistic ODE problem."""
    problem = logistic_ode()
    dynmod, measmod, initrv, info = problem

    times = np.arange(0, info["tmax"], info["dt"])
    obs = np.zeros((len(times), 1))

    states = info["ode"].solution(times)
    regression_problem = RegressionProblem(
        observations=obs, locations=times, solution=states
    )
    return dynmod, measmod, initrv, regression_problem


@pytest.fixture(params=[logistic_ode_problem])
def setup(request):
    """Filter and regression problem."""
    problem = request.param
    dynmod, measmod, initrv, regression_problem = problem()

    kalman = pnfs.Kalman(dynmod, measmod, initrv)
    return kalman, regression_problem


def test_rmse_filt_smooth(setup):
    """Assert that iterated smoothing beats smoothing beats filtering."""
    kalman, regression_problem = setup
    truth = regression_problem.solution

    stopcrit = pnfs.StoppingCriterion(atol=1e-1, rtol=1e-1, maxit=10)

    posterior = kalman.filter(regression_problem)
    posterior = kalman.smooth(posterior)

    iterated_posterior = kalman.iterated_filtsmooth(
        regression_problem, stopcrit=stopcrit
    )

    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean
    iterms = iterated_posterior.states.mean

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

    assert iterms_rmse < smooms_rmse
    assert smooms_rmse < filtms_rmse
