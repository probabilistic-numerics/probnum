import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth


@pytest.fixture(params=[filtsmooth_zoo.logistic_ode])
def setup(request):
    """Filter and regression problem."""
    problem = request.param
    regression_problem, info = problem()

    stopcrit = filtsmooth.optim.StoppingCriterion(atol=1e-1, rtol=1e-1, maxit=10)

    kalman = filtsmooth.gaussian.Kalman(
        info["prior_process"],
    )
    gauss_newton = filtsmooth.optim.GaussNewton(kalman, stopping_criterion=stopcrit)
    return gauss_newton, regression_problem


def test_rmse_filt_smooth(setup):
    """Assert that iterated smoothing beats smoothing."""

    np.random.seed(12345)
    gauss_newton, regression_problem = setup
    kalman = gauss_newton.kalman
    truth = regression_problem.solution

    posterior, _ = kalman.filter(regression_problem)
    posterior = kalman.smooth(posterior)

    iterated_posterior, _ = gauss_newton.solve(
        regression_problem, initial_guess=posterior
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
