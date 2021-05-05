import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth

# Problems


@pytest.fixture(params=[filtsmooth_zoo.car_tracking, filtsmooth_zoo.ornstein_uhlenbeck])
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


def test_kalman_smoother_high_order_ibm():
    """The highest feasible order (without damping) is 10.

    If this test breaks, someone played with the stable square-root
    implementations: for instance,
    """
    regression_problem, statespace_components = filtsmooth_zoo.car_tracking(
        model_ordint=10,
        timespan=(0.0, 1e-3),
        step=1e-6,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
    )
    truth = regression_problem.solution

    kalman = filtsmooth.Kalman(
        statespace_components["dynamics_model"],
        statespace_components["measurement_model"],
        statespace_components["initrv"],
    )

    posterior, _ = kalman.filtsmooth(regression_problem)

    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(regression_problem.observations - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse


def test_kalman_smoother_high_order_ibm_with_process_noise_damping_crazy_small_steps():
    """If we add a small damping factor to the computation of the Cholesky factor of the
    process noise covariance, we can do order 15 on the car tracking model at least."""
    regression_problem, statespace_components = filtsmooth_zoo.car_tracking(
        model_ordint=14,
        timespan=(0.0, 1e-10),
        step=1e-12,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
        _process_noise_damping=1e-14,
    )
    truth = regression_problem.solution

    kalman = filtsmooth.Kalman(
        statespace_components["dynamics_model"],
        statespace_components["measurement_model"],
        statespace_components["initrv"],
    )

    posterior, _ = kalman.filtsmooth(regression_problem)

    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(regression_problem.observations - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse


def test_kalman_smoother_high_order_ibm_with_process_noise_damping():
    """If we add a small damping factor to the computation of the Cholesky factor of the
    process noise covariance, we can do order 15 on the car tracking model at least."""
    regression_problem, statespace_components = filtsmooth_zoo.car_tracking(
        model_ordint=15,
        timespan=(0.0, 0.1),
        step=1e-3,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
        _process_noise_damping=1e-15,
    )
    truth = regression_problem.solution

    kalman = filtsmooth.Kalman(
        statespace_components["dynamics_model"],
        statespace_components["measurement_model"],
        statespace_components["initrv"],
    )

    posterior, _ = kalman.filtsmooth(regression_problem)

    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    smooms_rmse = np.mean(np.abs(smooms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(regression_problem.observations - truth[:, :2]))

    assert smooms_rmse < filtms_rmse < obs_rmse


def test_kalman_filter_high_order_ibm_with_process_noise_damping():
    """With damping, the filter achieves really high orders (we test 25, but 50 was
    possible too, for some reason)."""
    regression_problem, statespace_components = filtsmooth_zoo.car_tracking(
        model_ordint=25,
        timespan=(0.0, 1),
        step=1e-2,
        forward_implementation="sqrt",
        backward_implementation="sqrt",
        _process_noise_damping=1e-15,
    )
    truth = regression_problem.solution

    kalman = filtsmooth.Kalman(
        statespace_components["dynamics_model"],
        statespace_components["measurement_model"],
        statespace_components["initrv"],
    )

    posterior, _ = kalman.filter(regression_problem)

    filtms = posterior.states.mean

    filtms_rmse = np.mean(np.abs(filtms[:, :2] - truth[:, :2]))
    obs_rmse = np.mean(np.abs(regression_problem.observations - truth[:, :2]))

    assert filtms_rmse < obs_rmse
