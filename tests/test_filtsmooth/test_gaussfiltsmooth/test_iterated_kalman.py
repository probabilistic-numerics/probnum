import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth, problems


def logistic_ode_problem():
    """Logistic ODE problem."""
    logistic_ivp, statespace_components = filtsmooth_zoo.logistic_ode()

    times = np.arange(*logistic_ivp.timespan, step=0.2)
    obs = np.zeros((len(times), 1))

    states = logistic_ivp.solution(times)
    regression_problem = problems.RegressionProblem(
        observations=obs, locations=times, solution=states
    )
    return regression_problem, statespace_components


@pytest.fixture(params=[logistic_ode_problem])
def setup(request):
    """Filter and regression problem."""
    problem = request.param
    regression_problem, statespace_components = problem()

    kalman = filtsmooth.Kalman(
        statespace_components["dynamics_model"],
        statespace_components["measurement_model"],
        statespace_components["initrv"],
    )
    return kalman, regression_problem


def test_rmse_filt_smooth(setup):
    """Assert that iterated smoothing beats smoothing."""

    np.random.seed(12345)
    kalman, regression_problem = setup
    truth = regression_problem.solution

    stopcrit = filtsmooth.StoppingCriterion(atol=1e-1, rtol=1e-1, maxit=10)

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
    smooms_rmse = np.mean(np.abs(smooms[:, 0] - truth[:, 0]))
    iterms_rmse = np.mean(np.abs(iterms[:, 0] - truth[:, 0]))

    assert iterms_rmse < smooms_rmse
