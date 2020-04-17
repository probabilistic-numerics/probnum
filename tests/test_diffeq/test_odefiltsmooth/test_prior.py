"""
Test whether IBM(2) in 2 independent spatial dimensions
is as expected.


"""

import unittest

import numpy as np

from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.diffeq.odefiltsmooth import prior
from tests.testing import NumpyAssertions


STEP = np.random.rand()
DIFFCONST = np.random.rand()

AH_22_IBM = np.array([[1., STEP, STEP ** 2 / 2., 0., 0., 0.],
                  [0., 1., STEP, 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., STEP, STEP ** 2 / 2.],
                  [0., 0., 0., 0., 1., STEP],
                  [0., 0., 0., 0., 0., 1.]])

QH_22_IBM = DIFFCONST ** 2 * np.array(
    [[STEP ** 5 / 20., STEP ** 4 / 8., STEP ** 3 / 6., 0., 0., 0.],
     [STEP ** 4 / 8., STEP ** 3 / 3., STEP ** 2 / 2., 0., 0., 0.],
     [STEP ** 3 / 6., STEP ** 2 / 2., STEP, 0., 0., 0.],
     [0., 0., 0., STEP ** 5 / 20., STEP ** 4 / 8., STEP ** 3 / 6.],
     [0., 0., 0., STEP ** 4 / 8., STEP ** 3 / 3., STEP ** 2 / 2.],
     [0., 0., 0., STEP ** 3 / 6., STEP ** 2 / 2., STEP]])


class TestIBM(unittest.TestCase, NumpyAssertions):
    """
    """

    def setUp(self):
        """
        """
        self.ibm = prior.IBM(2, 2, DIFFCONST)

    def test_chapmankolmogorov(self):
        """
        """
        mean, cov = np.ones(self.ibm.ndim), np.eye(self.ibm.ndim)
        initdist = RandomVariable(distribution=Normal(mean, cov))
        cke, __ = self.ibm.chapmankolmogorov(0., STEP, STEP, initdist)
        self.assertAllClose(AH_22_IBM @ initdist.mean(), cke.mean(), 1e-14)
        self.assertAllClose(AH_22_IBM @ initdist.cov() @ AH_22_IBM.T + QH_22_IBM,
                            cke.cov(), 1e-14)


class TestIOUP(unittest.TestCase, NumpyAssertions):
    """
    """

    def setUp(self):
        """
        """
        driftspeed = np.random.rand()
        self.ibm = prior.IOUP(2, 2, driftspeed, DIFFCONST)

    def test_chapmankolmogorov(self):
        """
        """
        mean, cov = np.ones(self.ibm.ndim), np.eye(self.ibm.ndim)
        initdist = RandomVariable(distribution=Normal(mean, cov))
        self.ibm.chapmankolmogorov(0., STEP, STEP, initdist)


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
        """
        """
        lenscale, diffconst = np.random.rand(), np.random.rand()
        self.mat0 = prior.Matern(0, 1, lenscale, diffconst)
        self.mat1 = prior.Matern(1, 1, lenscale, diffconst)
        self.mat2 = prior.Matern(2, 1, lenscale, diffconst)

    def test_n0(self):
        """
        Closed form solution for n=0.
        This is OUP.
        """
        xi = np.sqrt(2*(self.mat0.ndim-0.5))/self.mat0.lengthscale
        self.assertAlmostEqual(self.mat0.driftmatrix[0, 0], -xi)

    def test_n1(self):
        """
        Closed form solution for n=1.
        """
        xi = np.sqrt(2*(self.mat1.ndim-0.5))/self.mat1.lengthscale
        expected = np.array([-xi**2, -2*xi])
        self.assertAllClose(self.mat1.driftmatrix[-1, :], expected)


    def test_n2(self):
        """
        Closed form solution for n=2.
        """
        xi = np.sqrt(2*(self.mat2.ndim-0.5))/self.mat2.lengthscale
        expected = np.array([-xi**3, -3*xi**2, -3*xi])
        self.assertAllClose(self.mat2.driftmatrix[-1, :], expected)

    def test_larger_shape(self):
        """
        """
        mat2d = prior.Matern(2, 2, 1., 1.)
        self.assertEqual(mat2d.ndim, 2*(2+1))


    def test_chapmankolmogorov(self):
        """
        """
        mean, cov = np.ones(self.mat1.ndim), np.eye(self.mat1.ndim)
        initdist = RandomVariable(distribution=Normal(mean, cov))
        self.mat1.chapmankolmogorov(0., STEP, STEP, initdist)

