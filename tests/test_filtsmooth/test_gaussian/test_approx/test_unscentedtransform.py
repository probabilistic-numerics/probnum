import unittest

import numpy as np

from probnum import filtsmooth, randvars


class TestUnscentedTransform(unittest.TestCase):
    def setUp(self):
        self.ndim = np.random.randint(1, 33)  # 1 < random int < 33
        alpha, beta, kappa = np.random.rand(3)
        self.ut = filtsmooth.gaussian.approx.UnscentedTransform(
            self.ndim, alpha, beta, kappa
        )
        self.mean = np.random.rand(self.ndim)
        cvr = np.random.rand(self.ndim, self.ndim)
        self.covar = cvr @ cvr.T

    def test_weights_shape(self):
        self.assertEqual(self.ut.mweights.ndim, 1)
        self.assertEqual(self.ut.mweights.shape[0], 2 * self.ndim + 1)
        self.assertEqual(self.ut.cweights.ndim, 1)
        self.assertEqual(self.ut.cweights.shape[0], 2 * self.ndim + 1)

    def test_sigpts_shape(self):
        sigpts = self.ut.sigma_points(randvars.Normal(self.mean, self.covar))
        self.assertEqual(sigpts.ndim, 2)
        self.assertEqual(sigpts.shape[0], 2 * self.ndim + 1)
        self.assertEqual(sigpts.shape[1], self.ndim)

    def test_propagate_shape(self):
        sigpts = self.ut.sigma_points(randvars.Normal(self.mean, self.covar))
        propagated = self.ut.propagate(None, sigpts, lambda t, x: np.sin(x))
        self.assertEqual(propagated.ndim, 2)
        self.assertEqual(propagated.shape[0], 2 * self.ndim + 1)
        self.assertEqual(propagated.shape[1], self.ndim)

    def test_estimate_statistics_shape(self):
        sigpts = self.ut.sigma_points(randvars.Normal(self.mean, self.covar))
        proppts = self.ut.propagate(None, sigpts, lambda t, x: np.sin(x))
        mest, cest, ccest = self.ut.estimate_statistics(
            proppts, sigpts, self.covar, self.mean
        )
        self.assertEqual(mest.ndim, 1)
        self.assertEqual(mest.shape[0], self.ndim)
        self.assertEqual(cest.ndim, 2)
        self.assertEqual(cest.shape[0], self.ndim)
        self.assertEqual(cest.shape[1], self.ndim)
        self.assertEqual(ccest.ndim, 2)
        self.assertEqual(ccest.shape[0], self.ndim)
        self.assertEqual(ccest.shape[1], self.ndim)

    def test_transform_of_gaussian_exact(self):
        sigpts = self.ut.sigma_points(randvars.Normal(self.mean, self.covar))
        ndim_meas = self.ndim + 1  # != self.ndim is important
        transmtrx = np.random.rand(ndim_meas, self.ndim)
        meascov = 0 * np.random.rand(ndim_meas, ndim_meas)
        proppts = self.ut.propagate(None, sigpts, lambda t, x: transmtrx @ x)
        mest, cest, ccest = self.ut.estimate_statistics(
            proppts, sigpts, meascov, self.mean
        )
        diff_mean = np.linalg.norm(mest - transmtrx @ self.mean)
        diff_covar = np.linalg.norm(cest - transmtrx @ self.covar @ transmtrx.T)
        diff_crosscovar = np.linalg.norm(ccest - self.covar @ transmtrx.T)
        self.assertLess(diff_mean / np.linalg.norm(transmtrx @ self.mean), 1e-11)
        self.assertLess(
            diff_covar / np.linalg.norm(transmtrx @ self.covar @ transmtrx.T), 1e-11
        )
        self.assertLess(
            diff_crosscovar / np.linalg.norm(self.covar @ transmtrx.T), 1e-11
        )
