import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth

# Problems


@pytest.fixture(params=[filtsmooth_zoo.car_tracking, filtsmooth_zoo.ornstein_uhlenbeck])
def setup(request, rng):
    """Filter and regression problem."""
    problem = request.param
    regression_problem, info = problem(rng=rng)

    kalman = filtsmooth.gaussian.Kalman(info["prior_process"])
    return kalman, regression_problem


def test_rmse_filt_smooth(setup):
    """Assert that smoothing beats filtering beats nothing."""

    np.random.seed(12345)
    kalman, regression_problem = setup
    truth = regression_problem.solution

    posterior, _ = kalman.filtsmooth(regression_problem)

    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(regression_problem.observations - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse


def test_info_dicts(setup):
    """Assert that smoothing beats filtering beats nothing."""

    np.random.seed(12345)
    kalman, regression_problem = setup

    posterior, info_dicts = kalman.filtsmooth(regression_problem)

    assert isinstance(info_dicts, list)
    assert len(posterior) == len(info_dicts)


def test_kalman_smoother_high_order_ibm(rng):
    """The highest feasible order (without damping, which we dont use) is 11.

    If this test breaks, someone played with the stable square-root implementations in
    discrete_transition: for instance, solve_triangular() and cho_solve() must not be
    changed to inv()!
    """
    regression_problem, info = filtsmooth_zoo.car_tracking(
        rng=rng,
        num_prior_derivatives=11,
        timespan=(0.0, 1e-3),
        step=1e-5,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    truth = regression_problem.solution

    kalman = filtsmooth.gaussian.Kalman(info["prior_process"])

    posterior, _ = kalman.filtsmooth(regression_problem)

    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(regression_problem.observations - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse


def test_kalman_multiple_measurement_models(rng):
    regression_problem, info = filtsmooth_zoo.car_tracking(
        rng=rng,
        num_prior_derivatives=4,
        timespan=(0.0, 1e-3),
        step=1e-5,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    truth = regression_problem.solution
    kalman = filtsmooth.gaussian.Kalman(info["prior_process"])

    posterior, _ = kalman.filtsmooth(regression_problem)

    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(regression_problem.observations - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse


def test_kalman_value_error_repeating_timepoints(rng):
    regression_problem, info = filtsmooth_zoo.car_tracking(
        rng=rng,
        num_prior_derivatives=4,
        timespan=(0.0, 1e-3),
        step=1e-5,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    kalman = filtsmooth.gaussian.Kalman(info["prior_process"])

    # This should raise a ValueError
    regression_problem.locations[1] = regression_problem.locations[0]

    with pytest.raises(ValueError):
        posterior, _ = kalman.filtsmooth(regression_problem)
