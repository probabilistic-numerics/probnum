"""

"""
import unittest
import numpy as np
from probnum.prob import RandomVariable, Normal
from probnum.prob.randomprocess import *


class NiceTestCases1d(unittest.TestCase):
    """
    """
    def setUp(self):
        """ """
        def rvmap(x):
            return RandomVariable(distribution=Normal(x, 0.1))

        self.supports = [None, np.array([-1.23 + 0.1*i for i in range(10)])]
        self.bounds = [None, (-np.inf, np.inf), (-4, 4)]
        self.rvseq = [RandomVariable(distribution=Normal())
                      for __ in range(10)]
        self.rvmap = rvmap

    def test_init_rvseq(self):
        """
        They should all raise ValueErrors
        """
        for supp in self.supports:
            for bds in self.bounds:
                with self.subTest(supp=supp, bds=bds):
                    RandomProcess(self.rvseq, support=supp, bounds=bds)

    def test_init_rvmap(self):
        """
        """
        for supp in self.supports:
            for bds in self.bounds:
                with self.subTest(supp=supp, bds=bds):
                    RandomProcess(self.rvmap, support=supp, bounds=bds)


class AdversarialTestCases(unittest.TestCase):
    """
    Parameter configurations that should lead to ValueErrors.
    """
    def setUp(self):
        """ """
        def rvmap(x):
            return RandomVariable(distribution=Normal(x, 0.1))

        self.supports = [2, np.ones(1), np.eye(3), "abc", [1, 2, 3]]
        self.bounds = [(-np.inf, 0), (1, 2, 3), [1, -1]]
        self.rvseq = np.array([RandomVariable(distribution=Normal())
                               for i in range(10)])
        self.rvmap = rvmap

    def test_init_rvseq(self):
        """
        They should all raise ValueErrors.
        """
        for supp in self.supports:
            for bds in self.bounds:
                with self.subTest(supp=supp, bds=bds):
                    with self.assertRaises(ValueError):
                        RandomProcess(self.rvseq, support=supp, bounds=bds)

    def test_init_rvmap(self):
        """
        They should all raise ValueErrors.
        """
        for supp in self.supports:
            for bds in self.bounds:
                with self.subTest(supp=supp, bds=bds):
                    with self.assertRaises(ValueError):
                        RandomProcess(self.rvmap, support=supp, bounds=bds)