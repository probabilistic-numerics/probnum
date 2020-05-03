"""

"""
import unittest
import numpy as np
from probnum.prob import RandomVariable, Normal
from probnum.prob.randomprocess import *
from tests.testing import NumpyAssertions

class NiceTestCases1d(unittest.TestCase):
    """
    """
    def setUp(self):
        """ """
        def rvmap(x):
            return RandomVariable(distribution=Normal(2.41, 0.1))

        self.supports = [None, np.array([-1.23 + 0.1*i for i in range(10)])]
        self.bounds = [None, (-np.inf, np.inf), (-4, 4)]
        self.rvseq = [RandomVariable(distribution=Normal())
                      for __ in range(10)]
        # self.rvseq = list(range(0, np.random.randint(10)))
        self.rvmap = rvmap

    def test_init_rvseq(self):
        """
        They should all raise ValueErrors
        """
        for supp in self.supports:
            with self.subTest(supp=supp, bds=None):
                RandomProcess(self.rvseq, supportpts=supp)

    def test_init_rvmap(self):
        """
        """
        for supp in self.supports:
            with self.subTest(supp=supp, bds=None):
                RandomProcess(self.rvmap, supportpts=supp)

        for bds in self.bounds:
            with self.subTest(supp=None, bds=bds):
                RandomProcess(self.rvmap, bounds=bds)


class ContinuousProcessTestCase(unittest.TestCase, NumpyAssertions):
    """
    Test case for continuous random processes.
    """
    def setUp(self):
        """ """
        def rvmap(x):
            return RandomVariable(distribution=Normal(2.41, 0.1))

        self.bounds = [None, (-np.inf, np.inf), (-400, 390),
                       [[-1, 1], [-2, 2]]]
        self.values = [1., 1., -395, [0.5, -1.5]]
        self.rvmap = rvmap

    def test_call(self):
        """ """
        for bds, vls in zip(self.bounds, self.values):
            with self.subTest(bds=bds):
                rp = RandomProcess(self.rvmap, bounds=bds)
                rv = rp(vls)
                self.assertIsInstance(rv, RandomVariable)

    def test_meanfun(self):
        """ """
        for bds, vls in zip(self.bounds, self.values):
            with self.subTest(bds=bds):
                rp = RandomProcess(self.rvmap, bounds=bds)
                mean = rp.meanfun(vls)
                self.assertEqual(mean, 2.41)

    def test_covfun(self):
        """ """
        for bds, vls in zip(self.bounds, self.values):
            with self.subTest(bds=bds):
                rp = RandomProcess(self.rvmap, bounds=bds)
                cov = rp.covfun(vls)
                self.assertEqual(cov, 0.1)

    def test_sample(self):
        """ """
        for bds, vls in zip(self.bounds, self.values):
            with self.subTest(bds=bds):
                rp = RandomProcess(self.rvmap, bounds=bds)
                rp.sample(vls)
                rp.sample(vls, size=10)
                rp.sample(vls, size=(10, 1, 2))

    def test_domain(self):
        """ """
        for bds, vls in zip(self.bounds, self.values):
            with self.subTest(bds=bds):
                rp = RandomProcess(self.rvmap, bounds=bds)
                # getter
                if bds is not None:
                    self.assertAllClose(rp.domain, bds)
                else:
                    real_line = np.array([-np.inf, np.inf])
                    self.assertAllClose(rp.domain, real_line)

                # setter and getter again
                if bds is not None:
                    rp.domain = 2*rp.domain
                    self.assertAllClose(rp.domain, 2*np.array(bds))

    def test_supportpts(self):
        """ """
        for bds, vls in zip(self.bounds, self.values):
            with self.subTest(bds=bds):
                rp = RandomProcess(self.rvmap, bounds=bds)
                # getter
                self.assertEqual(rp.supportpts, None)

                with self.assertRaises(NotImplementedError):
                    rp.supportpts = np.array(rp.supportpts)

    def test_bounds(self):
        """ """
        for bds, vls in zip(self.bounds, self.values):
            with self.subTest(bds=bds):
                rp = RandomProcess(self.rvmap, bounds=bds)
                # getter
                if bds is not None:
                    self.assertAllClose(rp.bounds, bds)
                else:
                    real_line = np.array([-np.inf, np.inf])
                    self.assertAllClose(rp.bounds, real_line)

                # setter and getter again
                if bds is not None:
                    rp.bounds = 2*rp.bounds
                    self.assertAllClose(rp.bounds, 2*np.array(bds))


class DiscreteProcessTestCase(unittest.TestCase):
    """
    Test case for continuous random processes.
    """
    pass