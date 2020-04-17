"""

"""
import unittest

import numpy as np
import scipy.linalg

from probnum.filtsmooth.statespace import util
from probnum.filtsmooth.gaussfiltsmooth import extendedkalman
from probnum.prob import RandomVariable, Dirac, Normal
from probnum.filtsmooth.statespace.continuous.linearsdemodel import *
from probnum.filtsmooth.statespace.discrete.discretegaussianmodel import *



np.random.seed(2532)
VISUALISE = False  # show plots or not?

if VISUALISE is True:
    import matplotlib.pyplot as plt

# Car tracking: Ex. 4.3 in Bayesian Filtering and Smoothing
DELTA_T = 0.1
VAR = 0.5
DYNAMAT = np.eye(4) + DELTA_T * np.diag(np.ones(2), 2)
DYNADIFF = np.diag(
    np.array([DELTA_T ** 3 / 3, DELTA_T ** 3 / 3, DELTA_T, DELTA_T])) \
           + np.diag(np.array([DELTA_T ** 2 / 2, DELTA_T ** 2 / 2]), 2) \
           + np.diag(np.array([DELTA_T ** 2 / 2, DELTA_T ** 2 / 2]), -2)
MEASMAT = np.eye(2, 4)
MEASDIFF = VAR * np.eye(2)
MEAN = np.zeros(4)
COV = 0.5 * VAR * np.eye(4)


class TestExtendedKalmanFilterDiscreteDiscrete(unittest.TestCase):
    """
    Try Kalman filtering on a discrete setting.
    """

    def setUp(self):
        """
        """
        self.dynmod = DiscreteGaussianLTIModel(DYNAMAT, np.zeros(len(DYNAMAT)),
                                      DYNADIFF)
        self.measmod = DiscreteGaussianLTIModel(MEASMAT, np.zeros(len(MEASMAT)),
                                       MEASDIFF)
        self.initdist = RandomVariable(distribution=Normal(MEAN, COV))
        self.kf = extendedkalman.ExtendedKalmanFilter(self.dynmod,
                                           self.measmod,
                                           self.initdist)

    def test_dynamicmodel(self):
        """
        """
        self.assertEqual(self.dynmod, self.kf.dynamicmodel)

    def test_measurementmodel(self):
        """
        """
        self.assertEqual(self.measmod, self.kf.measurementmodel)

    def test_initialdistribution(self):
        """
        """
        self.assertEqual(self.initdist, self.kf.initialdistribution)

    def test_predict(self):
        """
        """
        pred = self.kf.predict(0., DELTA_T, self.initdist)
        self.assertEqual(pred.mean().ndim, 1)
        self.assertEqual(pred.mean().shape[0], 4)
        self.assertEqual(pred.cov().ndim, 2)
        self.assertEqual(pred.cov().shape[0], 4)
        self.assertEqual(pred.cov().shape[1], 4)

    def test_update(self):
        """
        """
        data = self.measmod.sample(0., self.initdist.mean())
        data_as_rv = RandomVariable(distribution=Normal(data, np.zeros((len(data), len(data)))))
        upd, __, __, __ = self.kf.update(0., self.initdist, data_as_rv)
        self.assertEqual(upd.mean().ndim, 1)
        self.assertEqual(upd.mean().shape[0], 4)
        self.assertEqual(upd.cov().ndim, 2)
        self.assertEqual(upd.cov().shape[0], 4)
        self.assertEqual(upd.cov().shape[1], 4)

    def test_filter(self):
        """
        RMSE of filter smaller than rmse of measurements?
        """
        tms = np.arange(0, 20, DELTA_T)
        states, obs = util.generate_dd(self.dynmod, self.measmod,
                                             self.initdist, tms)
        means, covars = self.kf.filter(obs, tms)
        rmse_means = np.linalg.norm(means[1:, :2] - states[1:, :2]) / np.sqrt(
            states[1:, :2].size)
        rmse_obs = np.linalg.norm(obs - states[1:, :2]) / np.sqrt(
            states[1:, :2].size)
        if VISUALISE is True:
            plt.title("Car tracking trajectory (%.2f < %.2f?)" % (
                rmse_means, rmse_obs))
            plt.plot(obs[:, 0], obs[:, 1], '.', label="Observations",
                     alpha=0.5)
            plt.plot(means[:, 0], means[:, 1], '-', label="Filter guess")
            plt.plot(states[:, 0], states[:, 1], '-', linewidth=6, alpha=0.25,
                     label="Truth")
            plt.legend()
            plt.show()
        self.assertLess(rmse_means, rmse_obs)


class TestExtendedKalmanFilterContinuousDiscrete(unittest.TestCase):
    """
    Try Kalman filtering on a continuous-discrete setting.

    Try OU process.
    """

    def setUp(self):
        """
        """
        self.lam, self.q, r = 0.21, 0.5, 0.1
        self.drift = -self.lam * np.eye(1)
        self.force = np.zeros(1)
        self.disp = np.eye(1)
        self.diff = self.q * np.eye(1)
        self.dynmod = LTISDEModel(self.drift, self.force, self.disp, self.diff)
        self.measmod = DiscreteGaussianLTIModel(np.eye(1), np.zeros(1), r * np.eye(1))
        self.initdist = RandomVariable(distribution=Normal(10 * np.ones(1),
                                                      np.eye(1)))
        self.kf = extendedkalman.ExtendedKalmanFilter(self.dynmod,
                                           self.measmod,
                                           self.initdist)

    def test_dynamicmodel(self):
        """
        """
        self.assertEqual(self.dynmod, self.kf.dynamicmodel)

    def test_measurementmodel(self):
        """
        """
        self.assertEqual(self.measmod, self.kf.measurementmodel)

    def test_initialdistribution(self):
        """
        """
        self.assertEqual(self.initdist, self.kf.initialdistribution)

    def test_predict_shape(self):
        """
        """
        pred = self.kf.predict(0., DELTA_T, self.initdist)
        self.assertEqual(np.isscalar(pred.mean()), True)
        self.assertEqual(np.isscalar(pred.cov()), True)

    def test_predict_value(self):
        """
        """
        pred = self.kf.predict(0., DELTA_T, self.initdist)
        ah = scipy.linalg.expm(DELTA_T * self.drift)
        qh = self.q / (2 * self.lam) * (
                1 - scipy.linalg.expm(2 * self.drift * DELTA_T))
        diff_mean = ah @ (self.initdist.mean()*np.ones(1)) - pred.mean()*np.ones(1)
        diff_covar = ah @ (self.initdist.cov()*np.eye(1)) @ ah.T + qh - (pred.cov()*np.eye(1))
        self.assertLess(np.linalg.norm(diff_mean), 1e-14)
        self.assertLess(np.linalg.norm(diff_covar), 1e-14)

    def test_update(self):
        """
        """
        data = np.array([self.measmod.sample(0., self.initdist.mean()*np.ones(1))])
        data_as_rv = RandomVariable(distribution=Normal(data, np.zeros((len(data), len(data)))))
        upd, __, __, __ = self.kf.update(0., self.initdist, data_as_rv)
        self.assertEqual(np.isscalar(upd.mean()), True)
        self.assertEqual(np.isscalar(upd.cov()), True)

    def test_filter(self):
        """
        RMSE of filter smaller than rmse of measurements?
        """
        tms = np.arange(0, 20, DELTA_T)
        states, obs = util.generate_cd(self.dynmod, self.measmod,
                                             self.initdist, tms)
        means, covars = self.kf.filter(obs, tms)
        rmse_means = np.linalg.norm(means[1:] - states[1:]) / np.sqrt(
            states[1:].size)
        rmse_obs = np.linalg.norm(obs - states[1:]) / np.sqrt(states[1:].size)

        if VISUALISE is True:
            plt.title(
                "Ornstein Uhlenbeck (%.2f < %.2f?)" % (rmse_means, rmse_obs))
            plt.plot(tms[1:], obs[:, 0], '.', label="Observations", alpha=0.5)
            plt.plot(tms, means, '-', label="Filter guess")
            plt.plot(tms, states, '-', linewidth=6, alpha=0.25, label="Truth")
            plt.legend()
            plt.show()
        self.assertLess(rmse_means, rmse_obs)


class TestPendulum(unittest.TestCase):
    """
    We test on the pendulum example 5.1 in BFaS.
    """

    def setUp(self):
        """
        """
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
            y1 = ([1, delta_t])
            y2 = ([-g * np.cos(x1) * delta_t, 1])
            return np.array([y1, y2])

        def h(t, x):
            x1, x2 = x
            return np.array([np.sin(x1)])

        def dh(t, x):
            x1, x2 = x
            return np.array([[np.cos(x1), 0.]])

        q = 1.0 * (np.diag(np.array([delta_t ** 3 / 3, delta_t]))
                   + np.diag(np.array([delta_t ** 2 / 2]), 1)
                   + np.diag(np.array([delta_t ** 2 / 2]), -1))
        self.r = var * np.eye(1)
        initmean = np.ones(2)
        initcov = var * np.eye(2)
        self.dynamod = DiscreteGaussianModel(f, lambda t: q, df)
        self.measmod = DiscreteGaussianModel(h, lambda t: self.r, dh)
        self.initdist = RandomVariable(distribution=Normal(initmean, initcov))
        self.times = np.arange(0, 4, delta_t)
        self.q = q

    def test_filter(self):
        """
        """
        ukf = extendedkalman.ExtendedKalmanFilter(self.dynamod,
                                       self.measmod,
                                       self.initdist)
        states, obs = util.generate_dd(self.dynamod, self.measmod,
                                             self.initdist, self.times)
        means, covars = ukf.filter(obs, self.times)
        rmse_ukf = np.linalg.norm(means[:, 0] - states[:, 0]) / np.sqrt(
            means[:, 0].size)

        if VISUALISE is True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle("Noisy pendulum model (%.2f < %.2f?)" % (
                rmse_ukf, np.sqrt(self.r[0, 0])))
            ax1.set_title("Horizontal position")
            ax1.plot(self.times[1:], obs[:, 0], '.', alpha=0.25,
                     label="Observations")
            ax1.plot(self.times[1:], np.sin(states)[1:, 0], '-', linewidth=4,
                     alpha=0.5, label="Truth")
            ax1.plot(self.times[1:], np.sin(means)[1:, 0], '-', label="EKF")
            ax1.set_xlabel("time")
            ax1.set_ylabel("horizontal pos. = sin(angular)")
            ax1.legend()

            ax2.set_title("Angular position")
            # ax2.plot(self.times[1:], self.obs[:, 0], '.', alpha=0.25, label="Observations")
            ax2.plot(self.times[1:], (states)[1:, 0], '-', linewidth=4,
                     alpha=0.5, label="Truth")
            ax2.plot(self.times[1:], (means)[1:, 0], '-', label="EKF")
            ax2.set_xlabel("time")
            ax2.set_ylabel("angular pos.")
            ax2.legend()
            plt.show()

        self.assertLess(rmse_ukf, np.sqrt(self.r[0, 0]))
