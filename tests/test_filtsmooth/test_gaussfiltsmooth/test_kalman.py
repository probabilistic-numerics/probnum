""""""
import unittest

import numpy as np
import scipy.linalg

from probnum.filtsmooth.statespace import util
from probnum.filtsmooth.gaussfiltsmooth import kalman
from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.filtsmooth.statespace.continuous.linearsdemodel import *
from probnum.filtsmooth.statespace.discrete.discretegaussianmodel import *


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


class TestKalmanFilterDiscreteDiscrete(unittest.TestCase):
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
        self.kf = kalman.KalmanFilter(self.dynmod, self.measmod,
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
        upd, __, __, __ = self.kf.update(0., self.initdist, data)
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


class TestKalmanFilterContinuousDiscrete(unittest.TestCase):
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
        self.disp = np.ones(1)
        self.diff = self.q * np.eye(1)
        self.dynmod = LTISDEModel(self.drift, self.force, self.disp, self.diff)
        self.measmod = DiscreteGaussianLTIModel(np.eye(1), np.zeros(1), r * np.eye(1))
        self.initdist = RandomVariable(distribution=Normal(10 * np.ones(1), np.eye(1)))
        self.kf = kalman.KalmanFilter(self.dynmod, self.measmod, self.initdist)

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
        initmean = self.initdist.mean() * np.ones(1)
        initcov = self.initdist.cov() * np.eye(1)
        predmean = pred.mean() * np.ones(1)
        predcov = pred.cov() * np.eye(1)
        diff_mean = ah @ initmean - predmean
        diff_covar = ah @ initcov @ ah.T + qh - predcov
        self.assertLess(np.linalg.norm(diff_mean), 1e-14)
        self.assertLess(np.linalg.norm(diff_covar), 1e-14)

    def test_update(self):
        """
        """
        data = self.measmod.sample(0., self.initdist.mean()*np.ones(1))
        upd, __, __, __ = self.kf.update(0., self.initdist, data)
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
