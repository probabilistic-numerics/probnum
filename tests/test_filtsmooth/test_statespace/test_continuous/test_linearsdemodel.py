import unittest

import numpy as np

from probnum.random_variables import Normal
from probnum.filtsmooth.statespace.continuous import linearsdemodel

TEST_NDIM = 2


class TestLinearSDEModel(unittest.TestCase):
    """
    Check whether each function/property can be accessed
    as expected.
    """

    def setUp(self):
        self.driftmat = np.random.rand(TEST_NDIM, TEST_NDIM)
        self.dispmat = np.random.rand(TEST_NDIM)
        self.diffmat = self.driftmat @ self.driftmat.T + np.eye(TEST_NDIM)
        self.force = np.random.rand(TEST_NDIM)
        self.lm = linearsdemodel.LinearSDEModel(
            lambda t: self.driftmat,
            lambda t: self.force,
            lambda t: self.dispmat,
            self.diffmat,
        )

    def test_drift(self):
        some_state = np.random.rand(TEST_NDIM)
        diff = self.lm.drift(0.0, some_state) - (
            self.driftmat @ some_state + self.force
        )
        self.assertLess(np.linalg.norm(diff), 1e-14)

    def test_disp(self):
        some_state = np.random.rand(TEST_NDIM)
        diff = self.lm.dispersion(0.0, some_state) - self.dispmat
        self.assertLess(np.linalg.norm(diff), 1e-14)

    def test_jac(self):
        some_state = np.random.rand(TEST_NDIM)
        diff = self.lm.jacobian(0.0, some_state) - self.driftmat
        self.assertLess(np.linalg.norm(diff), 1e-14)

    def test_diff(self):
        diffusion = self.lm.diffusionmatrix
        self.assertLess(np.linalg.norm(diffusion - self.diffmat), 1e-14)

    def test_ndim(self):
        self.assertEqual(self.lm.ndim, TEST_NDIM)

    def test_sample(self):
        samp = self.lm.sample(0.0, 1.0, 0.1, (np.ones(TEST_NDIM)))
        self.assertEqual(samp.ndim, 1)
        self.assertEqual(samp.shape[0], TEST_NDIM)

    def test_chapmankolmogorov(self):
        """
        Test if CK-solution for a single step
        is according to iteration.
        """
        mean, cov = np.ones(TEST_NDIM), np.eye(TEST_NDIM)
        rvar = Normal(mean, cov)
        cke, __ = self.lm.chapmankolmogorov(0.0, 1.0, 1.0, rvar)
        diff_mean = self.driftmat @ rvar.mean + self.force - cke.mean + rvar.mean
        diff_cov = (
            self.driftmat @ rvar.cov
            + rvar.cov @ self.driftmat.T
            + self.dispmat @ self.diffmat @ self.dispmat.T
            + rvar.cov
            - cke.cov
        )
        self.assertLess(np.linalg.norm(diff_mean), 1e-14)
        self.assertLess(np.linalg.norm(diff_cov), 1e-14)


def ibm_a(step):
    """
    Closed for for A(h) for IBM prior.
    """
    return np.array([[1.0, step], [0.0, 1.0]])


def ibm_xi(step):
    """
    Closed for for \\xi(h) for IBM prior.
    """
    return np.zeros(2)


def ibm_q(step):
    """
    Closed for for Q(h) for IBM prior.
    """
    return 1.5 ** 2 * np.array([[step ** 3 / 3, step ** 2 / 2], [step ** 2 / 2, step]])


class TestLTISDEModel(unittest.TestCase):
    """
    Check whether each function/property can be accessed
    as expected.

    Test on Integrated Brownian motion (q=1, sig=1), driven by 1d BM,
    because it allows a closed form solution for integrals.
    """

    def setUp(self):
        self.driftmat = np.diag(np.ones(TEST_NDIM - 1), 1)
        self.dispmat = 1.5 * np.eye(TEST_NDIM)[:, -1].reshape((TEST_NDIM, 1))
        self.diffmat = np.eye(1)
        self.force = np.zeros(TEST_NDIM)
        self.lti = linearsdemodel.LTISDEModel(
            self.driftmat, self.force, self.dispmat, self.diffmat
        )

    def test_drift(self):
        some_state = np.random.rand(TEST_NDIM)
        diff = self.lti.drift(0.0, some_state) - (
            self.driftmat @ some_state + self.force
        )
        self.assertLess(np.linalg.norm(diff), 1e-14)

    def test_disp(self):
        some_state = np.random.rand(TEST_NDIM)
        diff = self.lti.dispersion(0.0, some_state) - self.dispmat
        self.assertLess(np.linalg.norm(diff), 1e-14)

    def test_jac(self):
        some_state = np.random.rand(TEST_NDIM)
        diff = self.lti.jacobian(0.0, some_state) - self.driftmat
        self.assertLess(np.linalg.norm(diff), 1e-14)

    def test_diff(self):
        diffusion = self.lti.diffusionmatrix
        self.assertLess(np.linalg.norm(diffusion - self.diffmat), 1e-14)

    def test_ndim(self):
        self.assertEqual(self.lti.ndim, TEST_NDIM)

    def test_sample(self):
        samp = self.lti.sample(0.0, 1.0, 0.1, (np.ones(TEST_NDIM)))
        self.assertEqual(samp.ndim, 1)
        self.assertEqual(samp.shape[0], TEST_NDIM)

    def test_driftmatrix(self):
        self.assertLess(np.linalg.norm(self.lti.driftmatrix - self.driftmat), 1e-14)

    def test_force(self):
        self.assertLess(np.linalg.norm(self.lti.force - self.force), 1e-14)

    def test_dispmatrix(self):
        self.assertLess(np.linalg.norm(self.lti.dispersionmatrix - self.dispmat), 1e-14)

    def test_chapmankolmogorov(self):
        """
        Test if CK-solution for a single step
        is according to closed form of IBM kernels..
        """
        mean, cov = np.ones(TEST_NDIM), np.eye(TEST_NDIM)
        rvar = Normal(mean, cov)
        delta = 0.1
        cke, __ = self.lti.chapmankolmogorov(0.0, 1.0, delta, rvar)
        ah, xih, qh = ibm_a(1.0), ibm_xi(1.0), ibm_q(1.0)
        diff_mean = np.linalg.norm(ah @ rvar.mean + xih - cke.mean)
        diff_cov = np.linalg.norm(ah @ rvar.cov @ ah.T + qh - cke.cov)
        self.assertLess(diff_mean, 1e-14)
        self.assertLess(diff_cov, 1e-14)
