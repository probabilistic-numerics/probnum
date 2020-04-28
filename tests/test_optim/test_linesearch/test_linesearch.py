""" 
"""

import unittest

import numpy as np

from probnum.optim import Objective, Eval
from probnum.optim import linesearch


class TestConstantLearningRate(unittest.TestCase):
    """
    """

    def test_next_lrate(self):
        """
        """
        constlr = linesearch.ConstantLearningRate(0.123456)
        self.assertEqual(constlr.next_lrate(None, None, None), 0.123456)

    def test_negative_lrate(self):
        """
        """
        with self.assertRaises(ValueError):
            linesearch.ConstantLearningRate(-1.0)

    def test_zero_lrate(self):
        """
        """
        with self.assertRaises(ValueError):
            linesearch.ConstantLearningRate(0.0)


class TestBacktrackingLineSearch(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """
        initguess, reduction = 1.0, 0.75
        self.backls = linesearch.BacktrackingLineSearch(initguess, reduction)

    def test_wrong_initguess(self):
        """
        """
        with self.assertRaises(ValueError):
            linesearch.BacktrackingLineSearch(initguess=0.0)
        with self.assertRaises(ValueError):
            linesearch.BacktrackingLineSearch(initguess=-1.0)
        with self.assertRaises(ValueError):
            linesearch.BacktrackingLineSearch(reduction=0.0)
        with self.assertRaises(ValueError):
            linesearch.BacktrackingLineSearch(reduction=-1.0)
        with self.assertRaises(ValueError):
            linesearch.BacktrackingLineSearch(reduction=2.0)
        with self.assertRaises(ValueError):
            linesearch.BacktrackingLineSearch(reduction=1.0)

    def test_direction_zero(self):
        """
        """

        def mockfct(x):
            return np.sum(x)

        mockeval = Eval(1.0, 1.0, 1.0, 1.0)
        direction = np.zeros(1)
        objec = Objective(mockfct)
        with self.assertRaises(ValueError):
            self.backls.next_lrate(mockeval, objec, direction)

    def test_initguess_good(self):
        """
        If initial guess is good enough, the loop should not be entered.

        E.g. for a quadratic problem with Newton iteration, lrate=1.0 is
        as good as possible.
        """

        def obj(x):
            return x.T @ x

        def grad(x):
            return 2 * x

        def hess(x):
            return 2 * np.ones((len(x), len(x)))

        objec = Objective(obj, grad, hess)
        mockeval = Eval(np.array([10.0]), 100.0,
                                  np.array([20.0]), np.array([2.0]))
        lrate = self.backls.next_lrate(mockeval, objec, np.array([-10.0]))
        self.assertEqual(lrate, 1.0)

    def test_lrate_condition(self):
        """
        Computes a learning rate and checks whether it leads
        to a clear reduction in function value.
        """

        def obj(x):
            return x.T @ x

        def grad(x):
            return 2 * x

        objec = Objective(obj, grad)
        iteration = Eval(np.array([10.0]), 100,
                                   np.array([20.0]), None)
        direction = np.array([-1000.0])  # should yield small lrate!
        lrate = self.backls.next_lrate(iteration, objec, direction)
        proposed = objec.objective(iteration.x + lrate * direction)
        comparison = iteration.fx + 0.5 * lrate * iteration.dfx @ direction
        self.assertLess(proposed, comparison)
