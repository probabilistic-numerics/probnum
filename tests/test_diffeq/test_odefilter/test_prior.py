"""
Test whether IBM(2) in 2 independent spatial dimensions
is as expected.


"""

import unittest

import numpy as np

from probnum.prob import RandomVariable
from probnum.prob.distributions import Normal
from probnum.diffeq.odefilter import prior

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


class TestIBM(unittest.TestCase):
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
        cke = self.ibm.chapmankolmogorov(0., STEP, STEP, initdist)
        diff_mean = np.linalg.norm(AH_22_IBM @ initdist.mean() - cke.mean())
        diff_covar = np.linalg.norm(
            AH_22_IBM @ initdist.cov() @ AH_22_IBM.T + QH_22_IBM - cke.cov())
        self.assertLess(diff_mean, 1e-14)
        self.assertLess(diff_covar, 1e-14)


class TestIOUP(unittest.TestCase):
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

class TestMatern(unittest.TestCase):
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
        diff_0 = np.abs(self.mat0.driftmatrix[0, 0] + xi)
        self.assertLess(diff_0, 1e-14)


    def test_n1(self):
        """
        Closed form solution for n=1.
        """
        xi = np.sqrt(2*(self.mat1.ndim-0.5))/self.mat1.lengthscale
        diff_0 = np.abs(self.mat1.driftmatrix[-1, 0] + xi**2)
        diff_1 = np.abs(self.mat1.driftmatrix[-1, 1] + 2*xi)
        self.assertLess(diff_0, 1e-14)
        self.assertLess(diff_1, 1e-14)

    def test_n2(self):
        """
        Closed form solution for n=2.
        """
        xi = np.sqrt(2*(self.mat2.ndim-0.5))/self.mat2.lengthscale
        diff_0 = np.abs(self.mat2.driftmatrix[-1, 0] + xi**3)
        diff_1 = np.abs(self.mat2.driftmatrix[-1, 1] + 3*xi**2)
        diff_2 = np.abs(self.mat2.driftmatrix[-1, 2] + 3*xi)
        self.assertLess(diff_0, 1e-14)
        self.assertLess(diff_1, 1e-14)
        self.assertLess(diff_2, 1e-14)

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

