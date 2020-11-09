"""
Test cases for Gaussian Filtering and Smoothing
"""
import unittest

import matplotlib.pyplot as plt
import numpy as np

import probnum.filtsmooth as pnfs
from probnum.random_variables import Normal
from tests.testing import NumpyAssertions

__all__ = [
    "CarTrackingDDTestCase",
    "OrnsteinUhlenbeckCDTestCase",
    "LinearisedDiscreteTransitionTestCase",
]


def car_tracking():
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

    dynmod = pnfs.statespace.DiscreteLTIGaussian(
        dynamat=dynamat, forcevec=np.zeros(4), diffmat=dynadiff
    )
    measmod = pnfs.statespace.DiscreteLTIGaussian(
        dynamat=measmat, forcevec=np.zeros(2), diffmat=measdiff
    )
    initrv = Normal(mean, cov)
    return dynmod, measmod, initrv, {"dt": delta_t}


class CarTrackingDDTestCase(unittest.TestCase, NumpyAssertions):
    """
    Car tracking: Ex. 4.3 in Bayesian Filtering and Smoothing
    """

    def setup_cartracking(self):
        self.dynmod, self.measmod, self.initrv, info = car_tracking()
        self.delta_t = info["dt"]
        self.tms = np.arange(0, 20, self.delta_t)
        self.states, self.obs = pnfs.statespace.generate(
            self.dynmod, self.measmod, self.initrv, self.tms
        )


def ornstein_uhlenbeck():
    delta_t = 0.2
    lam, q, r = 0.21, 0.5, 0.1
    drift = -lam * np.eye(1)
    force = np.zeros(1)
    disp = np.sqrt(q) * np.eye(1)
    dynmod = pnfs.statespace.LTISDE(
        driftmatrix=drift,
        forcevec=force,
        dispmatrix=disp,
    )
    measmod = pnfs.statespace.DiscreteLTIGaussian(
        dynamat=np.eye(1), forcevec=np.zeros(1), diffmat=r * np.eye(1)
    )
    initrv = Normal(10 * np.ones(1), np.eye(1))
    return dynmod, measmod, initrv, {"dt": delta_t}


class OrnsteinUhlenbeckCDTestCase(unittest.TestCase, NumpyAssertions):
    """
    Ornstein Uhlenbeck process as a test case.
    """

    def setup_ornsteinuhlenbeck(self):
        self.dynmod, self.measmod, self.initrv, info = ornstein_uhlenbeck()
        self.delta_t = info["dt"]
        self.tms = np.arange(0, 20, self.delta_t)
        self.states, self.obs = pnfs.statespace.generate(
            dynmod=self.dynmod, measmod=self.measmod, initrv=self.initrv, times=self.tms
        )


def pendulum():
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
    dynamod = pnfs.statespace.DiscreteGaussian(f, lambda t: q, df)
    measmod = pnfs.statespace.DiscreteGaussian(h, lambda t: r, dh)
    initrv = Normal(initmean, initcov)
    return dynamod, measmod, initrv, {"dt": delta_t}


class LinearisedDiscreteTransitionTestCase(unittest.TestCase, NumpyAssertions):
    """
    Test approximate Gaussian filtering and smoothing

    1. Transition RV is enabled by linearising
    2. Applied to a linear model, the outcome is exact
    3. Smoothing RMSE < Filtering RMSE < Data RMSE on the pendulum example.
    """

    # overwrite by implementation
    visualise = False

    def test_transition_rv(self):
        """transition_rv() not possible for original model but for the linearised model"""
        nonlinear_model, _, initrv, _ = pendulum()
        linearised_model = self.linearising_component_pendulum(nonlinear_model)

        with self.subTest("Baseline should not work."):
            with self.assertRaises(NotImplementedError):
                nonlinear_model.transition_rv(initrv, 0.0)
        with self.subTest("Linearisation happens."):
            linearised_model.transition_rv(initrv, 0.0)

    def test_exactness_linear_model(self):
        """Applied to a linear model, the results should be unchanged."""
        linear_model, _, initrv, _ = car_tracking()
        linearised_model = self.linearising_component_car(linear_model)

        with self.subTest("Different objects"):
            self.assertNotIsInstance(linear_model, type(linearised_model))

        received, info1 = linear_model.transition_rv(initrv, 0.0)
        expected, info2 = linearised_model.transition_rv(initrv, 0.0)
        crosscov1 = info1["crosscov"]
        crosscov2 = info2["crosscov"]
        rtol, atol = 1e-10, 1e-10
        with self.subTest("Same outputs"):
            self.assertAllClose(received.mean, expected.mean, rtol=rtol, atol=atol)
            self.assertAllClose(received.cov, expected.cov, rtol=rtol, atol=atol)
            self.assertAllClose(crosscov1, crosscov2, rtol=rtol, atol=atol)

    def test_filtsmooth_pendulum(self):

        # Set up test problem
        self.dynamod, self.measmod, self.initrv, info = pendulum()
        delta_t = info["dt"]
        self.tms = np.arange(0, 4, delta_t)
        self.states, self.obs = pnfs.statespace.generate(
            self.dynamod, self.measmod, self.initrv, self.tms
        )

        # Linearise problem
        self.ekf_meas = self.linearising_component_pendulum(self.measmod)
        self.ekf_dyna = self.linearising_component_pendulum(self.dynamod)
        self.method = pnfs.Kalman(self.ekf_dyna, self.ekf_meas, self.initrv)

        # Compute filter/smoother solution
        filter_posterior = self.method.filter(self.obs, self.tms)
        filtms = filter_posterior.state_rvs.mean
        smooth_posterior = self.method.filtsmooth(self.obs, self.tms)
        smooms = smooth_posterior.state_rvs.mean

        # Compute RMSEs
        comp = self.states[:, 0]
        normaliser = np.sqrt(comp.size)
        filtrmse = np.linalg.norm(filtms[:, 0] - comp) / normaliser
        smoormse = np.linalg.norm(smooms[:, 0] - comp) / normaliser
        obs_rmse = np.linalg.norm(self.obs[:, 0] - comp[1:]) / normaliser

        # If desired, visualise.
        if self.visualise is True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(
                "Noisy pendulum model (%.2f " % smoormse
                + "< %.2f < %.2f?)" % (filtrmse, obs_rmse)
            )
            ax1.set_title("Horizontal position")
            ax1.plot(
                self.tms[1:], self.obs[:, 0], ".", alpha=0.25, label="Observations"
            )
            ax1.plot(
                self.tms[1:],
                np.sin(self.states)[1:, 0],
                "-",
                linewidth=4,
                alpha=0.5,
                label="Truth",
            )
            ax1.plot(self.tms[1:], np.sin(filtms)[1:, 0], "-", label="Filter")
            ax1.plot(self.tms[1:], np.sin(smooms)[1:, 0], "-", label="Smoother")
            ax1.set_xlabel("time")
            ax1.set_ylabel("horizontal pos. = sin(angular)")
            ax1.legend()

            ax2.set_title("Angular position")
            ax2.plot(
                self.tms[1:],
                self.states[1:, 0],
                "-",
                linewidth=4,
                alpha=0.5,
                label="Truth",
            )
            ax2.plot(self.tms[1:], filtms[1:, 0], "-", label="Filter")
            ax2.plot(self.tms[1:], smooms[1:, 0], "-", label="Smoother")
            ax2.set_xlabel("time")
            ax2.set_ylabel("angular pos.")
            ax2.legend()
            plt.show()

        # Test if RMSEs behave well.
        self.assertLess(smoormse, filtrmse)
        self.assertLess(filtrmse, obs_rmse)
