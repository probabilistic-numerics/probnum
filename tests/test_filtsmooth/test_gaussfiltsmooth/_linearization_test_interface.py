"""Test interface for EKF and UKF."""

import numpy as np
import pytest

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth, problems, randprocs, randvars


class InterfaceTestDiscreteLinearization:
    """Test approximate Gaussian filtering and smoothing.

    1. Transition RV is enabled by linearising
    2. Applied to a linear model, the outcome is exact
    3. Smoothing RMSE < Filtering RMSE < Data RMSE on the pendulum example.
    """

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.linearizing_component = None
        self.linearizing_function_regression_problem = None

    def test_transition_rv(self):
        """forward_rv() not possible for original model but for the linearised model."""
        # pylint: disable=not-callable
        _, statespace_components = filtsmooth_zoo.pendulum()
        non_linear_model = statespace_components["dynamics_model"]
        initrv = statespace_components["initrv"]
        linearised_model = self.linearizing_component(non_linear_model)

        # Baseline: non-linear model should not work
        with pytest.raises(NotImplementedError):
            non_linear_model.forward_rv(initrv, 0.0)

        # Linearized model works
        rv, _ = linearised_model.forward_rv(initrv, 0.0)
        assert isinstance(rv, randvars.RandomVariable)

    def test_exactness_linear_model(self):
        """Applied to a linear model, the results should be unchanged."""
        # pylint: disable=not-callable
        regression_problem, statespace_components = filtsmooth_zoo.car_tracking()
        linear_model = statespace_components["dynamics_model"]
        initrv = statespace_components["initrv"]
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

    def test_filtsmooth_pendulum(self):
        # pylint: disable=not-callable
        # Set up test problem

        # If this measurement variance is not really small, the sampled
        # test data can contain an outlier every now and then which
        # breaks the test, even though it has not been touched.
        regression_problem, statespace_components = filtsmooth_zoo.pendulum(
            measurement_variance=0.0001
        )

        ekf_dyna = self.linearizing_component(statespace_components["dynamics_model"])

        regression_problem = problems.TimeSeriesRegressionProblem(
            locations=regression_problem.locations,
            observations=regression_problem.observations,
            measurement_models=statespace_components["measurement_model"],
            solution=regression_problem.solution,
        )
        linearized_problem = self.linearizing_function_regression_problem(
            regression_problem
        )

        initrv = statespace_components["initrv"]
        prior_process = randprocs.MarkovProcess(
            transition=ekf_dyna, initrv=initrv, initarg=regression_problem.locations[0]
        )
        method = filtsmooth.Kalman(prior_process)

        # Compute filter/smoother solution
        posterior, _ = method.filtsmooth(linearized_problem)
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
