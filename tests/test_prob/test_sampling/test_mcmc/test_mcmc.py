"""
Unittest for the convenience functions in mcmc.py

Essesntially, a rewriting of the doctests in mcmc.py

We only check the shapes of the outputs. More elaborate tests, e.g. qq-plots,
are done in the corresponding modules.
"""

import unittest
import numpy as np
from probnum.prob.sampling.mcmc.mcmc import *

VISUALIZE = False    # set to True if you'd like to see hist plots

if VISUALIZE is True:
    import matplotlib.pyplot as plt


np.random.seed(98687513)


class MCMCTestCase(unittest.TestCase):
    """
    Testcase for MCMC tests.

    We test only the convenience functions, see the corresponding
    modules for more.

    """
    def setUp(self):
        """ """
        def logpdf(x):
            return x.T @ x / 2.0

        def logder(x):
            return x

        def loghess(x):
            return np.ones((len(x), len(x)))

        xval = np.linspace(-4, 4, 200).reshape((200, 1))
        yval = np.array([np.exp(-logpdf(x)) / np.sqrt(2*np.pi) for x in xval])

        self.logpdf = logpdf
        self.logder = logder
        self.loghess = loghess
        self.xval = xval
        self.yval = yval


class TestRandomWalk(MCMCTestCase):
    """ """

    def test_rwmh(self):
        """ """

        states, lklds, accratio = rwmh(self.logpdf, 7500, np.array([.5]), pwidth=18.0)
        self.assertEqual(states.ndim, 2)
        self.assertEqual(states.shape[0], 7500)
        self.assertEqual(states.shape[1], 1)
        self.assertEqual(lklds.ndim, 1)
        self.assertEqual(lklds.shape[0], 7500)
        self.assertEqual(np.isscalar(accratio), True)
        self.assertLess(0., accratio)
        self.assertLess(accratio, 1.)

        if VISUALIZE is True:
            __ = plt.plot(self.xval, self.yval)
            __ = plt.hist(states[:, 0], bins=50, density=True, alpha=0.5)
            __ = plt.title("RW Samples")
            plt.show()


class TestMALA(MCMCTestCase):
    """ """

    def test_mala(self):
        """ """

        states, lklds, accratio = mala(self.logpdf, self.logder, 2500, np.array([.5]), pwidth=1.5)
        self.assertEqual(states.ndim, 2)
        self.assertEqual(states.shape[0], 2500)
        self.assertEqual(states.shape[1], 1)
        self.assertEqual(lklds.ndim, 1)
        self.assertEqual(lklds.shape[0], 2500)
        self.assertEqual(np.isscalar(accratio), True)
        self.assertLess(0., accratio)
        self.assertLess(accratio, 1.)

        if VISUALIZE is True:
            __ = plt.plot(self.xval, self.yval)
            __ = plt.hist(states[:, 0], bins=50, density=True, alpha=0.5)
            __ = plt.title("MALA Samples")
            plt.show()

    def test_pmala(self):
        """ """

        states, lklds, accratio = pmala(self.logpdf, self.logder, self.loghess, 2500, np.array([.5]), pwidth=1.5)
        self.assertEqual(states.ndim, 2)
        self.assertEqual(states.shape[0], 2500)
        self.assertEqual(states.shape[1], 1)
        self.assertEqual(lklds.ndim, 1)
        self.assertEqual(lklds.shape[0], 2500)
        self.assertEqual(np.isscalar(accratio), True)
        self.assertLess(0., accratio)
        self.assertLess(accratio, 1.)

        if VISUALIZE is True:
            __ = plt.plot(self.xval, self.yval)
            __ = plt.hist(states[:, 0], bins=50, density=True, alpha=0.5)
            __ = plt.title("PMALA Samples")
            plt.show()




class TestHMC(MCMCTestCase):
    """ """

    def test_hmc(self):
        """ """

        states, lklds, accratio = hmc(self.logpdf, self.logder, 2500, initstate=np.array([.5]), stepsize=1.75, nsteps=5)
        self.assertEqual(states.ndim, 2)
        self.assertEqual(states.shape[0], 2500)
        self.assertEqual(states.shape[1], 1)
        self.assertEqual(lklds.ndim, 1)
        self.assertEqual(lklds.shape[0], 2500)
        self.assertEqual(np.isscalar(accratio), True)
        self.assertLess(0., accratio)
        self.assertLess(accratio, 1.)

        if VISUALIZE is True:
            __ = plt.plot(self.xval, self.yval)
            __ = plt.hist(states[:, 0], bins=50, density=True, alpha=0.5)
            __ = plt.title("HMC Samples")
            plt.show()

    def test_phmc(self):
        """ """

        states, lklds, accratio = phmc(self.logpdf, self.logder, self.loghess, 2500, initstate=np.array([.5]), stepsize=1.75, nsteps=5)
        self.assertEqual(states.ndim, 2)
        self.assertEqual(states.shape[0], 2500)
        self.assertEqual(states.shape[1], 1)
        self.assertEqual(lklds.ndim, 1)
        self.assertEqual(lklds.shape[0], 2500)
        self.assertEqual(np.isscalar(accratio), True)
        self.assertLess(0., accratio)
        self.assertLess(accratio, 1.)

        if VISUALIZE is True:
            __ = plt.plot(self.xval, self.yval)
            __ = plt.hist(states[:, 0], bins=50, density=True, alpha=0.5)
            __ = plt.title("PHMC Samples")
            plt.show()

