"""Test cases for Gaussian Filtering and Smoothing."""
import unittest

import numpy as np

import probnum.problems.zoo.filtsmooth as filtsmooth_zoo
from probnum import filtsmooth
from tests.testing import NumpyAssertions

__all__ = [
    "LinearisedDiscreteTransitionTestCase",
]

# Show plots in tests?
VISUALISE = False

if VISUALISE:
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "Install matplotlib to visualise the test functions."
        ) from err


np.random.seed(12345)


class LinearisedDiscreteTransitionTestCase(unittest.TestCase, NumpyAssertions):
    """Test approximate Gaussian filtering and smoothing.

    1. Transition RV is enabled by linearising
    2. Applied to a linear model, the outcome is exact
    3. Smoothing RMSE < Filtering RMSE < Data RMSE on the pendulum example.
    """

    linearising_component_pendulum = NotImplemented
    linearising_component_car = NotImplemented

    def test_transition_rv(self):
        """forward_rv() not possible for original model but for the linearised model."""
        # pylint: disable=not-callable
        _, statespace_components = filtsmooth_zoo.pendulum()
        nonlinear_model = statespace_components["dynamics_model"]
        initrv = statespace_components["initrv"]
        linearised_model = self.linearising_component_pendulum(nonlinear_model)

        with self.subTest("Baseline should not work."):
            with self.assertRaises(NotImplementedError):
                nonlinear_model.forward_rv(initrv, 0.0)
        with self.subTest("Linearisation happens."):
            linearised_model.forward_rv(initrv, 0.0)

    def test_exactness_linear_model(self):
        """Applied to a linear model, the results should be unchanged."""
        # pylint: disable=not-callable
        regression_problem, statespace_components = filtsmooth_zoo.car_tracking()
        linear_model = statespace_components["dynamics_model"]
        initrv = statespace_components["initrv"]
        linearised_model = self.linearising_component_car(linear_model)

        with self.subTest("Different objects"):
            self.assertNotIsInstance(linear_model, type(linearised_model))

        received, info1 = linear_model.forward_rv(initrv, 0.0)
        expected, info2 = linearised_model.forward_rv(initrv, 0.0)
        crosscov1 = info1["crosscov"]
        crosscov2 = info2["crosscov"]
        rtol, atol = 1e-10, 1e-10
        with self.subTest("Same outputs"):
            self.assertAllClose(received.mean, expected.mean, rtol=rtol, atol=atol)
            self.assertAllClose(received.cov, expected.cov, rtol=rtol, atol=atol)
            self.assertAllClose(crosscov1, crosscov2, rtol=rtol, atol=atol)

    def test_filtsmooth_pendulum(self):
        # pylint: disable=not-callable
        # Set up test problem
        regression_problem, statespace_components = filtsmooth_zoo.pendulum()

        # Linearise problem
        ekf_meas = self.linearising_component_pendulum(
            statespace_components["measurement_model"]
        )
        ekf_dyna = self.linearising_component_pendulum(
            statespace_components["dynamics_model"]
        )
        method = filtsmooth.Kalman(ekf_dyna, ekf_meas, statespace_components["initrv"])

        # Compute filter/smoother solution
        posterior, _ = method.filtsmooth(regression_problem)
        filtms = posterior.filtering_posterior.states.mean
        smooms = posterior.states.mean

        # Compute RMSEs
        comp = regression_problem.solution[:, 0]
        normaliser = np.sqrt(comp.size)
        filtrmse = np.linalg.norm(filtms[:, 0] - comp) / normaliser
        smoormse = np.linalg.norm(smooms[:, 0] - comp) / normaliser
        obs_rmse = (
            np.linalg.norm(regression_problem.observations[:, 0] - comp) / normaliser
        )

        # If desired, visualise.
        if VISUALISE:
            obs, tms, states = (
                regression_problem.observations,
                regression_problem.locations,
                regression_problem.solution,
            )
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(
                "Noisy pendulum model (%.2f " % smoormse
                + "< %.2f < %.2f?)" % (filtrmse, obs_rmse)
            )
            ax1.set_title("Horizontal position")
            ax1.plot(tms[1:], obs[:, 0], ".", alpha=0.25, label="Observations")
            ax1.plot(
                tms[1:],
                np.sin(states)[1:, 0],
                "-",
                linewidth=4,
                alpha=0.5,
                label="Truth",
            )
            ax1.plot(tms[1:], np.sin(filtms)[1:, 0], "-", label="Filter")
            ax1.plot(tms[1:], np.sin(smooms)[1:, 0], "-", label="Smoother")
            ax1.set_xlabel("time")
            ax1.set_ylabel("horizontal pos. = sin(angular)")
            ax1.legend()

            ax2.set_title("Angular position")
            ax2.plot(
                tms[1:],
                states[1:, 0],
                "-",
                linewidth=4,
                alpha=0.5,
                label="Truth",
            )
            ax2.plot(tms[1:], filtms[1:, 0], "-", label="Filter")
            ax2.plot(tms[1:], smooms[1:, 0], "-", label="Smoother")
            ax2.set_xlabel("time")
            ax2.set_ylabel("angular pos.")
            ax2.legend()
            plt.show()

        # Test if RMSEs behave well.
        self.assertLess(smoormse, filtrmse)
        self.assertLess(filtrmse, obs_rmse)


class LinearisedContinuousTransitionTestCase(unittest.TestCase, NumpyAssertions):
    """Test approximate Gaussian filtering and smoothing.

    1. Transition RV is enabled by linearising
    2. Applied to a linear model, the outcome is exact
    3. Smoothing RMSE < Filtering RMSE < Data RMSE on the Benes-Daum example.
    """

    linearising_component_benes_daum = NotImplemented

    def test_transition_rv(self):
        """forward_rv() not possible for original model but for the linearised model."""
        # pylint: disable=not-callable
        _, statespace_components = filtsmooth_zoo.benes_daum()
        nonlinear_model = statespace_components["dynamics_model"]
        initrv = statespace_components["initrv"]
        linearised_model = self.linearising_component_benes_daum(nonlinear_model)

        with self.subTest("Baseline should not work."):
            with self.assertRaises(NotImplementedError):
                nonlinear_model.forward_rv(initrv, 0.0, 1.0)
        with self.subTest("Linearisation happens."):
            linearised_model.forward_rv(initrv, 0.0, 1.0)

    def test_transition_real(self):
        """transition_real() not possible for original model but for the linearised
        model."""
        # pylint: disable=not-callable
        _, statespace_components = filtsmooth_zoo.benes_daum()
        nonlinear_model = statespace_components["dynamics_model"]
        initrv = statespace_components["initrv"]
        linearised_model = self.linearising_component_benes_daum(nonlinear_model)

        with self.subTest("Baseline should not work."):
            with self.assertRaises(NotImplementedError):
                nonlinear_model.forward_realization(initrv.mean, 0.0, 1.0)
        with self.subTest("Linearisation happens."):
            linearised_model.forward_realization(initrv.mean, 0.0, 1.0)
