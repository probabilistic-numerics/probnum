import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth


@pytest.fixture(params=[filtsmooth_zoo.logistic_ode])
def setup(request):
    """Filter and regression problem."""
    problem = request.param
    regression_problem, info = problem()

    kalman = filtsmooth.Kalman(
        info["prior_process"],
    )
    return (kalman, regression_problem)


def test_rmse_filt_smooth(setup):
    """Assert that iterated smoothing beats smoothing."""

    np.random.seed(12345)
    kalman, regression_problem = setup
    truth = regression_problem.solution

    stopcrit = filtsmooth.StoppingCriterion(atol=1e-1, rtol=1e-1, maxit=10)

    posterior, _ = kalman.filter(regression_problem)
    posterior = kalman.smooth(posterior)

    iterated_posterior, _ = kalman.iterated_filtsmooth(
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
