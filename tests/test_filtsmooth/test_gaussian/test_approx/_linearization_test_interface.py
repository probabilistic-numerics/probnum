"""Test interface for EKF and UKF."""

import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth, problems, randprocs, randvars


class InterfaceDiscreteLinearizationTest:
    """Test approximate Gaussian filtering and smoothing.

    1. forward_rv is unlocked by linearization
    2. Applied to a linear model, the outcome is exactly the same as the original transition.
    3. Smoothing RMSE < Filtering RMSE < Data RMSE on the pendulum example.
    """

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.linearizing_component = None

    def test_transition_rv(self, rng):
        """forward_rv() not possible for original model but for the linearised model."""
        # pylint: disable=not-callable
        _, info = filtsmooth_zoo.pendulum(rng=rng)
        non_linear_model = info["prior_process"].transition
        initrv = info["prior_process"].initrv
        linearised_model = self.linearizing_component(non_linear_model)

        # Baseline: non-linear model should not work
        with pytest.raises(NotImplementedError):
            non_linear_model.forward_rv(initrv, 0.0)

        # Linearized model works
        rv, _ = linearised_model.forward_rv(initrv, 0.0)
        assert isinstance(rv, randvars.RandomVariable)

    def test_exactness_linear_model(self, rng):
        """Applied to a linear model, the results should be unchanged."""
        # pylint: disable=not-callable
        regression_problem, info = filtsmooth_zoo.car_tracking(rng=rng)
        linear_model = info["prior_process"].transition
        initrv = info["prior_process"].initrv
        linearised_model = self.linearizing_component(linear_model)

        # Assert that the objects are different
        assert not isinstance(linear_model, type(linearised_model))

        # Assert that the give the same outputs.
        received, info1 = linear_model.forward_rv(initrv, 0.0)
        expected, info2 = linearised_model.forward_rv(initrv, 0.0)
        crosscov1 = info1["crosscov"]
        crosscov2 = info2["crosscov"]
        rtol, atol = 1e-10, 1e-10
        np.testing.assert_allclose(received.mean, expected.mean, rtol=rtol, atol=atol)
        np.testing.assert_allclose(received.cov, expected.cov, rtol=rtol, atol=atol)
        np.testing.assert_allclose(crosscov1, crosscov2, rtol=rtol, atol=atol)

    def test_filtsmooth_pendulum(self, rng):
        # pylint: disable=not-callable
        # Set up test problem

        # If this measurement variance is not really small, the sampled
        # test data can contain an outlier every now and then which
        # breaks the test, even though it has not been touched.
        regression_problem, info = filtsmooth_zoo.pendulum(
            rng=rng, measurement_variance=0.0001
        )
        prior_process = info["prior_process"]
        measmods = regression_problem.measurement_models

        ekf_dyna = self.linearizing_component(prior_process.transition)
        ekf_meas = [self.linearizing_component(mm) for mm in measmods]

        regression_problem = problems.TimeSeriesRegressionProblem(
            locations=regression_problem.locations,
            observations=regression_problem.observations,
            measurement_models=ekf_meas,
            solution=regression_problem.solution,
        )

        initrv = prior_process.initrv
        prior_process = randprocs.markov.MarkovProcess(
            transition=ekf_dyna, initrv=initrv, initarg=regression_problem.locations[0]
        )
        method = filtsmooth.gaussian.Kalman(prior_process)

        # Compute filter/smoother solution
        posterior, _ = method.filtsmooth(regression_problem)
        filtms = posterior.filtering_posterior.states.mean
        smooms = posterior.states.mean

        # Compute RMSEs and assert they are well-behaved.
        comp = regression_problem.solution[:, 0]
        normaliser = np.sqrt(comp.size)
        filtrmse = np.linalg.norm(filtms[:, 0] - comp) / normaliser
        smoormse = np.linalg.norm(smooms[:, 0] - comp) / normaliser
        obs_rmse = (
            np.linalg.norm(regression_problem.observations[:, 0] - comp) / normaliser
        )

        assert smoormse < filtrmse < obs_rmse


class InterfaceContinuousLinearizationTest:
    """Interface for tests of approximate, nonlinear Gaussian filtering and
    smoothing."""

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.linearizing_component = None

    def test_transition_rv(self, rng):
        """forward_rv() not possible for original model but for the linearised model."""
        # pylint: disable=not-callable
        _, info = filtsmooth_zoo.benes_daum(rng=rng)
        prior_process = info["prior_process"]
        non_linear_model = prior_process.transition
        initrv = prior_process.initrv
        linearized_model = self.linearizing_component(non_linear_model)

        # Baseline: non-linear model should not work
        with pytest.raises(NotImplementedError):
            non_linear_model.forward_rv(initrv, t=0.0, dt=0.1)

        # Linearized model works
        rv, _ = linearized_model.forward_rv(initrv, t=0.0, dt=0.1)
        assert isinstance(rv, randvars.RandomVariable)

    def test_filtsmooth_benes_daum(self, rng):
        # pylint: disable=not-callable
        # Set up test problem

        # If this measurement variance is not really small, the sampled
        # test data can contain an outlier every now and then which
        # breaks the test, even though it has not been touched.
        time_grid = np.arange(0.0, 5.0, step=0.1)

        regression_problem, info = filtsmooth_zoo.benes_daum(
            rng=rng, measurement_variance=1e-1, time_grid=time_grid
        )
        prior_process = info["prior_process"]
        ekf_dyna = self.linearizing_component(prior_process.transition)
        initrv = prior_process.initrv
        prior_process = randprocs.markov.MarkovProcess(
            transition=ekf_dyna, initrv=initrv, initarg=regression_problem.locations[0]
        )
        method = filtsmooth.gaussian.Kalman(prior_process)

        # Compute filter/smoother solution
        posterior, _ = method.filter(regression_problem)
        posterior = method.smooth(posterior)
        filtms = posterior.filtering_posterior.states.mean
        smooms = posterior.states.mean

        # Compute RMSEs and assert they are well-behaved.
        comp = regression_problem.solution[:, 0]
        normaliser = np.sqrt(comp.size)
        filtrmse = np.linalg.norm(filtms[:, 0] - comp) / normaliser
        smoormse = np.linalg.norm(smooms[:, 0] - comp) / normaliser
        obs_rmse = (
            np.linalg.norm(regression_problem.observations[:, 0] - comp) / normaliser
        )
        assert smoormse < filtrmse < obs_rmse
