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
