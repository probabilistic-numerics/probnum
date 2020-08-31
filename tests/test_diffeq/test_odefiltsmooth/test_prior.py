"""
Tests include:

- IBM recovers closed form solutions of Chapman-Kolmogorov equations.
- IOUP is same as IBM for driftspeed = 0.0
- Matern driftmatrix satisfies closed form solutions.
"""

import unittest

import numpy as np

from probnum.random_variables import Normal
from probnum.diffeq.odefiltsmooth import prior
from tests.testing import NumpyAssertions


STEP = np.random.rand()
DIFFCONST = np.random.rand()

AH_22_IBM = np.array(
    [
        [1.0, STEP, STEP ** 2 / 2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, STEP, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, STEP, STEP ** 2 / 2.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, STEP],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)

QH_22_IBM = DIFFCONST ** 2 * np.array(
    [
        [STEP ** 5 / 20.0, STEP ** 4 / 8.0, STEP ** 3 / 6.0, 0.0, 0.0, 0.0],
        [STEP ** 4 / 8.0, STEP ** 3 / 3.0, STEP ** 2 / 2.0, 0.0, 0.0, 0.0],
        [STEP ** 3 / 6.0, STEP ** 2 / 2.0, STEP, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, STEP ** 5 / 20.0, STEP ** 4 / 8.0, STEP ** 3 / 6.0],
        [0.0, 0.0, 0.0, STEP ** 4 / 8.0, STEP ** 3 / 3.0, STEP ** 2 / 2.0],
        [0.0, 0.0, 0.0, STEP ** 3 / 6.0, STEP ** 2 / 2.0, STEP],
    ]
)


AH_21_PRE = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])

QH_21_PRE = (
    DIFFCONST ** 2
    * STEP
    * np.array([[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1]])
)


class TestIBM(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.prior = prior.IBM(2, 2, DIFFCONST)

    def test_chapmankolmogorov(self):
        mean, cov = np.ones(self.prior.ndim), np.eye(self.prior.ndim)
        initrv = Normal(mean, cov)
        cke, __ = self.prior.chapmankolmogorov(0.0, STEP, STEP, initrv)
        self.assertAllClose(AH_22_IBM @ initrv.mean, cke.mean, 1e-14)
        self.assertAllClose(
            AH_22_IBM @ initrv.cov @ AH_22_IBM.T + QH_22_IBM, cke.cov, 1e-14
        )

    def test_chapmankolmogorov_super_comparison(self):
        """
        The result of chapmankolmogorov() should be identical to the matrix fraction decomposition technique
        implemented in LinearSDE, just faster.
        """
        # pylint: disable=bad-super-call
        mean, cov = np.ones(self.prior.ndim), np.eye(self.prior.ndim)
        initrv = Normal(mean, cov)
        cke_super, __ = super(type(self.prior), self.prior).chapmankolmogorov(
            0.0, STEP, STEP, initrv
        )
        cke, __ = self.prior.chapmankolmogorov(0.0, STEP, STEP, initrv)

        self.assertAllClose(cke_super.mean, cke.mean, 1e-14)
        self.assertAllClose(cke_super.cov, cke.cov, 1e-14)


class TestIBMPrecond(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        self.prior = prior.IBM(
            ordint=2, spatialdim=1, diffconst=DIFFCONST, precond_step=STEP
        )

    def test_chapmankolmogorov(self):
        mean, cov = np.ones(self.prior.ndim), np.eye(self.prior.ndim)
        initrv = Normal(mean, cov)
        cke, __ = self.prior.chapmankolmogorov(0.0, STEP, STEP, initrv)

        self.assertAllClose(AH_21_PRE @ initrv.mean, cke.mean, 1e-14)
        self.assertAllClose(
            AH_21_PRE @ initrv.cov @ AH_21_PRE.T + QH_21_PRE, cke.cov, 1e-14
        )

    def test_chapmankolmogorov_super_comparison(self):
        """
        The result of chapmankolmogorov() should be identical to the matrix fraction decomposition technique
        implemented in LinearSDE, just faster.
        """
        # pylint: disable=bad-super-call

        mean, cov = np.ones(self.prior.ndim), np.eye(self.prior.ndim)
        initrv = Normal(mean, cov)
        cke_super, __ = super(type(self.prior), self.prior).chapmankolmogorov(
            0.0, STEP, STEP, initrv
        )
        cke, __ = self.prior.chapmankolmogorov(0.0, STEP, STEP, initrv)

        self.assertAllClose(cke_super.mean, cke.mean, 1e-14)
        self.assertAllClose(cke_super.cov, cke.cov, 1e-14)


class TestIOUP(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        driftspeed = np.random.rand()
        self.ibm = prior.IOUP(2, 2, driftspeed, DIFFCONST)

    def test_chapmankolmogorov(self):
        mean, cov = np.ones(self.ibm.ndim), np.eye(self.ibm.ndim)
        initrv = Normal(mean, cov)
        self.ibm.chapmankolmogorov(0.0, STEP, STEP, initrv)

    def test_asymptotically_ibm(self):
        """
        Checks that for driftspeed==0, it coincides with the IBM prior.
        """
        ioup_speed0 = prior.IOUP(2, 3, driftspeed=0, diffconst=1.2345)
        ibm = prior.IBM(2, 3, diffconst=1.2345)
        self.assertAllClose(ioup_speed0.driftmatrix, ibm.driftmatrix)
        self.assertAllClose(ioup_speed0.dispersionmatrix, ibm.dispersionmatrix)
        self.assertAllClose(ioup_speed0.diffusionmatrix, ibm.diffusionmatrix)


class TestMatern(unittest.TestCase, NumpyAssertions):
    """
    Test whether coefficients for q=1, 2 match closed form.
    and whether coefficients for q=0 are Ornstein Uhlenbeck.
    """

    def setUp(self):
        lenscale, diffconst = np.random.rand(), np.random.rand()
        self.mat0 = prior.Matern(0, 1, lenscale, diffconst)
        self.mat1 = prior.Matern(1, 1, lenscale, diffconst)
        self.mat2 = prior.Matern(2, 1, lenscale, diffconst)

    def test_n0(self):
        """
        Closed form solution for n=0.
        This is OUP.
        """
        xi = np.sqrt(2 * (self.mat0.ndim - 0.5)) / self.mat0.lengthscale
        self.assertAlmostEqual(self.mat0.driftmatrix[0, 0], -xi)

    def test_n1(self):
        """
        Closed form solution for n=1.
        """
        xi = np.sqrt(2 * (self.mat1.ndim - 0.5)) / self.mat1.lengthscale
        expected = np.array([-(xi ** 2), -2 * xi])
        self.assertAllClose(self.mat1.driftmatrix[-1, :], expected)

    def test_n2(self):
        """
        Closed form solution for n=2.
        """
        xi = np.sqrt(2 * (self.mat2.ndim - 0.5)) / self.mat2.lengthscale
        expected = np.array([-(xi ** 3), -3 * xi ** 2, -3 * xi])
        self.assertAllClose(self.mat2.driftmatrix[-1, :], expected)

    def test_larger_shape(self):
        mat2d = prior.Matern(2, 2, 1.0, 1.0)
        self.assertEqual(mat2d.ndim, 2 * (2 + 1))

    def test_chapmankolmogorov(self):
        mean, cov = np.ones(self.mat1.ndim), np.eye(self.mat1.ndim)
        initrv = Normal(mean, cov)
        self.mat1.chapmankolmogorov(0.0, STEP, STEP, initrv)
