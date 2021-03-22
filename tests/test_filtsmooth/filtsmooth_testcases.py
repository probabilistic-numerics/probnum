"""Test cases for Gaussian Filtering and Smoothing."""
import unittest

import numpy as np

import probnum.diffeq as pnd  # ODE problem as test function
import probnum.filtsmooth as pnfs
import probnum.statespace as pnss
from probnum.randvars import Constant, Normal
from tests.testing import NumpyAssertions

__all__ = [
    "CarTrackingDDTestCase",
    "OrnsteinUhlenbeckCDTestCase",
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


def car_tracking():

    # Below is for consistency with pytest & unittest.
    # Without a seed, unittest passes but pytest fails.
    # I tried multiple seeds, they all work equally well.
    np.random.seed(12345)

    delta_t = 0.2
    var = 0.5
    dynamat = np.eye(4) + delta_t * np.diag(np.ones(2), 2)
    dynadiff = (
        np.diag(np.array([delta_t ** 3 / 3, delta_t ** 3 / 3, delta_t, delta_t]))
        + np.diag(np.array([delta_t ** 2 / 2, delta_t ** 2 / 2]), 2)
        + np.diag(np.array([delta_t ** 2 / 2, delta_t ** 2 / 2]), -2)
    )
    measmat = np.eye(2, 4)
    measdiff = var * np.eye(2)
    mean = np.zeros(4)
    cov = 0.5 * var * np.eye(4)

    dynmod = pnss.DiscreteLTIGaussian(
        state_trans_mat=dynamat, shift_vec=np.zeros(4), proc_noise_cov_mat=dynadiff
    )
    measmod = pnss.DiscreteLTIGaussian(
        state_trans_mat=measmat,
        shift_vec=np.zeros(2),
        proc_noise_cov_mat=measdiff,
    )
    initrv = Normal(mean, cov)
    return dynmod, measmod, initrv, {"dt": delta_t, "tmax": 20}


class CarTrackingDDTestCase(unittest.TestCase, NumpyAssertions):
    """Car tracking: Ex.

    4.3 in Bayesian Filtering and Smoothing
    """

    def setup_cartracking(self):
        self.dynmod, self.measmod, self.initrv, info = car_tracking()
        self.delta_t = info["dt"]
        self.tms = np.arange(0, 20, self.delta_t)
        self.states, self.obs = pnss.generate_samples(
            self.dynmod, self.measmod, self.initrv, self.tms
        )


def ornstein_uhlenbeck():

    # Below is for consistency with pytest & unittest.
    # Without a seed, unittest passes but pytest fails.
    # I tried multiple seeds, they all work equally well.
    np.random.seed(12345)

    delta_t = 0.2
    lam, q, r = 0.21, 0.5, 0.1
    drift = -lam * np.eye(1)
    force = np.zeros(1)
    disp = np.sqrt(q) * np.eye(1)
    dynmod = pnss.LTISDE(
        driftmat=drift,
        forcevec=force,
        dispmat=disp,
    )
    measmod = pnss.DiscreteLTIGaussian(
        state_trans_mat=np.eye(1),
        shift_vec=np.zeros(1),
        proc_noise_cov_mat=r * np.eye(1),
    )
    initrv = Normal(10 * np.ones(1), np.eye(1))
    return dynmod, measmod, initrv, {"dt": delta_t, "tmax": 20}


class OrnsteinUhlenbeckCDTestCase(unittest.TestCase, NumpyAssertions):
    """Ornstein Uhlenbeck process as a test case."""

    def setup_ornsteinuhlenbeck(self):
        self.dynmod, self.measmod, self.initrv, info = ornstein_uhlenbeck()
        self.delta_t = info["dt"]
        self.tmax = info["tmax"]
        self.tms = np.arange(0, self.tmax, self.delta_t)
        self.states, self.obs = pnss.generate_samples(
            dynmod=self.dynmod, measmod=self.measmod, initrv=self.initrv, times=self.tms
        )


def pendulum():

    # Below is for consistency with pytest & unittest.
    # Without a seed, unittest passes but pytest fails.
    # I tried multiple seeds, they all work equally well.
    np.random.seed(12345)

    delta_t = 0.0075
    var = 0.32 ** 2
    g = 9.81

    def f(t, x):
        x1, x2 = x
        y1 = x1 + x2 * delta_t
        y2 = x2 - g * np.sin(x1) * delta_t
        return np.array([y1, y2])

    def df(t, x):
        x1, x2 = x
        y1 = [1, delta_t]
        y2 = [-g * np.cos(x1) * delta_t, 1]
        return np.array([y1, y2])

    def h(t, x):
        x1, x2 = x
        return np.array([np.sin(x1)])

    def dh(t, x):
        x1, x2 = x
        return np.array([[np.cos(x1), 0.0]])

    q = 1.0 * (
        np.diag(np.array([delta_t ** 3 / 3, delta_t]))
        + np.diag(np.array([delta_t ** 2 / 2]), 1)
        + np.diag(np.array([delta_t ** 2 / 2]), -1)
    )
    r = var * np.eye(1)
    initmean = np.ones(2)
    initcov = var * np.eye(2)
    dynamod = pnss.DiscreteGaussian(2, 2, f, lambda t: q, df)
    measmod = pnss.DiscreteGaussian(2, 1, h, lambda t: r, dh)
    initrv = Normal(initmean, initcov)
    return dynamod, measmod, initrv, {"dt": delta_t, "tmax": 4}


def logistic_ode():

    # Below is for consistency with pytest & unittest.
    # Without a seed, unittest passes but pytest fails.
    # I tried multiple seeds, they all work equally well.
    np.random.seed(12345)
    delta_t = 0.2
    tmax = 2

    logistic = pnd.logistic((0, tmax), initrv=Constant(np.array([0.1])), params=(6, 1))
    dynamod = pnss.IBM(ordint=3, spatialdim=1)
    measmod = pnfs.DiscreteEKFComponent.from_ode(
        logistic, dynamod, np.zeros((1, 1)), ek0_or_ek1=1
    )

    initmean = np.array([0.1, 0, 0.0, 0.0])
    initcov = np.diag([0.0, 1.0, 1.0, 1.0])
    initrv = Normal(initmean, initcov)

    return dynamod, measmod, initrv, {"dt": delta_t, "tmax": tmax, "ode": logistic}


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
        nonlinear_model, _, initrv, _ = pendulum()
        linearised_model = self.linearising_component_pendulum(nonlinear_model)

        with self.subTest("Baseline should not work."):
            with self.assertRaises(NotImplementedError):
                nonlinear_model.forward_rv(initrv, 0.0)
        with self.subTest("Linearisation happens."):
            linearised_model.forward_rv(initrv, 0.0)

    def test_exactness_linear_model(self):
        """Applied to a linear model, the results should be unchanged."""
        # pylint: disable=not-callable
        linear_model, _, initrv, _ = car_tracking()
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
        dynamod, measmod, initrv, info = pendulum()
        delta_t = info["dt"]
        tmax = info["tmax"]
        tms = np.arange(0, tmax, delta_t)
        states, obs = pnss.generate_samples(dynamod, measmod, initrv, tms)

        # Linearise problem
        ekf_meas = self.linearising_component_pendulum(measmod)
        ekf_dyna = self.linearising_component_pendulum(dynamod)
        method = pnfs.Kalman(ekf_dyna, ekf_meas, initrv)

        # Compute filter/smoother solution
        posterior = method.filtsmooth(obs, tms)
        filtms = posterior.filtering_posterior.state_rvs.mean
        smooms = posterior.state_rvs.mean

        # Compute RMSEs
        comp = states[:, 0]
        normaliser = np.sqrt(comp.size)
        filtrmse = np.linalg.norm(filtms[:, 0] - comp) / normaliser
        smoormse = np.linalg.norm(smooms[:, 0] - comp) / normaliser
        obs_rmse = np.linalg.norm(obs[:, 0] - comp) / normaliser

        # If desired, visualise.
        if VISUALISE:
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


def benes_daum():
    """Benes-Daum testcase, example 10.17 in Applied SDEs."""

    def f(t, x):
        return np.tanh(x)

    def df(t, x):
        return 1.0 - np.tanh(x) ** 2

    def l(t):
        return np.ones(1)

    initmean = np.zeros(1)
    initcov = 3.0 * np.eye(1)
    initrv = Normal(initmean, initcov)
    dynamod = pnss.SDE(dimension=1, driftfun=f, dispmatfun=l, jacobfun=df)
    measmod = pnss.DiscreteLTIGaussian(np.eye(1), np.zeros(1), np.eye(1))
    return dynamod, measmod, initrv, {}


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
        nonlinear_model, _, initrv, _ = benes_daum()
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
        nonlinear_model, _, initrv, _ = benes_daum()
        linearised_model = self.linearising_component_benes_daum(nonlinear_model)

        with self.subTest("Baseline should not work."):
            with self.assertRaises(NotImplementedError):
                nonlinear_model.forward_realization(initrv.mean, 0.0, 1.0)
        with self.subTest("Linearisation happens."):
            linearised_model.forward_realization(initrv.mean, 0.0, 1.0)
