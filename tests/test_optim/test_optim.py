"""
Unittest for the convenience functions in optim.py

Essesntially, a rewriting of the doctests in optim.py

"""


import unittest
import numpy as np
from probnum.optim.optim import *

class OptimTestCase(unittest.TestCase):
    """
    """
    def setUp(self):
        """ """
        def obj(x):
            return x.T @ x

        def der(x):
            return 2.0 * x

        def hess(x):
            return 2.0 * np.ones((len(x), len(x)))

        self.obj = obj
        self.der = der
        self.hess = hess


class TestRandomSearch(OptimTestCase):
    """
    Test for the facade in optim.py
    """
    def test_minimise_rs(self):
        """ """
        initval = np.array([15.0])
        traj, obs = minimise_rs(self.obj, initval, tol=0.01, lrate=0.65, maxit=150)
        self.assertAlmostEqual(traj[-1][0], 0.05, places=2)


class TestGradientDescent(OptimTestCase):
    """
    Test for the facade in optim.py
    """
    def test_lrate(self):
        """ """
        initval = np.array([15.0])
        traj, obs = minimise_gd(self.obj, self.der, initval, tol=1e-05, lrate=0.75)
        self.assertAlmostEqual(traj[-1][0], 0., places=4)

    def test_lsearch(self):
        """ """
        initval = np.array([15.0])
        traj, obs = minimise_gd(self.obj, self.der, initval, tol=1e-05, lsearch="backtrack")
        self.assertAlmostEqual(traj[-1][0], 0., places=4)


class TestNewton(OptimTestCase):
    """
    Test for the facade in optim.py
    """
    def test_lrate(self):
        """ """
        initval = np.array([15.0])
        traj, obs = minimise_newton(self.obj, self.der, self.hess, initval, tol=1e-06, lrate=0.75)
        self.assertAlmostEqual(traj[-1][0], 0., places=4)

    def test_lsearch(self):
        """ """
        initval = np.array([15.0])
        traj, obs = minimise_newton(self.obj, self.der, self.hess, initval, tol=1e-06, lsearch="backtrack")
        self.assertAlmostEqual(traj[-1][0], 0., places=4)


class TestLevMarq(OptimTestCase):
    """
    Test for the facade in optim.py
    """
    def test_lrate(self):
        """ """
        initval = np.array([15.0])
        traj, obs = minimise_levmarq(self.obj, self.der, self.hess, initval, dampingpar=0.1, tol=1e-06, lrate=0.75)
        self.assertAlmostEqual(traj[-1][0], 0., places=4)

    def test_lsearch(self):
        """ """
        initval = np.array([15.0])
        traj, obs = minimise_levmarq(self.obj, self.der, self.hess, initval, dampingpar=0.1, tol=1e-06, lsearch="backtrack")
        self.assertAlmostEqual(traj[-1][0], 0., places=4)