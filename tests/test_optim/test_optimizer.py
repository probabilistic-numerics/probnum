"""
Unittests for optimiser class.

Included
--------
* minimise_nd complains about non-ndarray initval
* minimise_nd complains if objective does not
  evaluate to scalar
"""

import unittest

import numpy as np

from probnum.optim import optimizer, objective
from probnum.optim import linesearch, stoppingcriterion


class MockOptimizer(optimizer.Optimizer):
    """
    Mock object to check whether Optimiser can be subclassed
    """

    def iterate(self, curriter, **kwargs):
        """
        Print optional arguments. Used for testing
        whether these are passed down properly.
        """
        return curriter


class TestOptimizer(unittest.TestCase):
    """
    Tests whether Optimiser class can be subclassed
    successfully, even with an empty iterate() method.
    """

    def setUp(self):
        """
        Will fail if subclass didn't work properly.
        """

        def obj(x):
            return x.T @ x

        def grad(x):
            return 2 * x

        self.obj = objective.Objective(obj, grad)
        stopcrit = stoppingcriterion.DiffOfFctValues(0.0)
        lsearch = linesearch.ConstantLearningRate(0.1)
        self.opt = MockOptimizer(lsearch, stopcrit, maxit=2)

    def test_objective_not_scalar(self):
        """
        If objective is not a scalar, it should raise a TypeError.
        """

        def obj(x):
            return np.array(x)

        objec = objective.Objective(obj)
        with self.assertRaises(TypeError):
            self.opt.minimise_nd(objec, np.array([100.0, 200.0]))

    def test_initval_no_array(self):
        """
        If initval is not a ndarray, it should raise a TypeError.
        """
        with self.assertRaises(TypeError):
            self.opt.minimise_nd(self.obj, 200.0)
