import numpy as np
import pytest
import pytest_cases

from probnum import filtsmooth, problems, randprocs
import probnum.problems.zoo.filtsmooth as filtsmooth_zoo


def case_logistic_ode():
    """Filter and regression problem."""
    regression_problem, info = filtsmooth_zoo.logistic_ode()
    kalman = filtsmooth.gaussian.ContinuousKalman(
        info["prior_process"],
    )
    return kalman, regression_problem


def case_pendulum():
    """Filter and regression problem."""
    rng = np.random.default_rng(seed=1)
    regression_problem, info = filtsmooth_zoo.pendulum(rng=rng)

    # Linearize measurement models
    measmods = regression_problem.measurement_models
    ekf_meas = [filtsmooth.gaussian.approx.DiscreteEKFComponent(mm) for mm in measmods]
    regression_problem = problems.TimeSeriesRegressionProblem(
        locations=regression_problem.locations,
        observations=regression_problem.observations,
        measurement_models=ekf_meas,
        solution=regression_problem.solution,
    )

    # Linearize prior
    prior_process = info["prior_process"]
    ekf_dyna = filtsmooth.gaussian.approx.DiscreteEKFComponent(prior_process.transition)
    prior_process_linearized = randprocs.markov.MarkovProcess(
        transition=ekf_dyna,
        initrv=prior_process.initrv,
        initarg=prior_process.initarg,
    )

    # Assemble Kalman filter
    kalman = filtsmooth.gaussian.DiscreteKalman(prior_process_linearized)
    return kalman, regression_problem


@pytest_cases.parametrize_with_cases("kalman, regression_problem", cases=".")
def test_rmse_filt_smooth(kalman, regression_problem):
    """Assert that iterated smoothing beats smoothing."""

    # Assemble iterated smoother
    stopcrit = filtsmooth.optim.FiltSmoothStoppingCriterion(
        atol=1e-1, rtol=1e-1, maxit=7
    )
    gauss_newton = filtsmooth.optim.GaussNewton(kalman, stopping_criterion=stopcrit)

    # Compute filtering, smoothing, and iterated smoothing solutions
    posterior, _ = kalman.filter(regression_problem)
    posterior = kalman.smooth(posterior)
    iterated_posterior, _ = gauss_newton.solve(
        regression_problem, initial_guess=posterior
    )
    filtms = posterior.filtering_posterior.states.mean
    smooms = posterior.states.mean
    iterms = iterated_posterior.states.mean

    # Compare only zeroth component
    # for compatibility with all test cases
    truth = regression_problem.solution
    if truth.ndim == 1:
        truth = truth.reshape((-1, 1))
    if filtms.ndim == 1:
        smooms = smooms.reshape((-1, 1))
        iterms = iterms.reshape((-1, 1))
    smooms_rmse = np.mean(np.abs(smooms[:, 0] - truth[:, 0]))
    iterms_rmse = np.mean(np.abs(iterms[:, 0] - truth[:, 0]))

    assert iterms_rmse < smooms_rmse
