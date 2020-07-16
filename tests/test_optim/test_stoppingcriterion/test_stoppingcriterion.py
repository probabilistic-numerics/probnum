"""

Included
--------
* AbsoluteTolerance:
    - make a fulfilled case, check if fulfilled(...) returns True
    - make a non-fulfilled case, check if fulfilled(...) returns False
    - fulfilled(...) complains about nonexisting
      values for curriter.dfx
    - create_unfulfilled creates something where fulfilled(...) returns True 
"""

import unittest

from probnum.optim import objective
from probnum.optim import stoppingcriterion


class TestNormOfGradient(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """
        self.tol = 1e-4
        self.stopcrit = stoppingcriterion.NormOfGradient(self.tol)

    def test_fulfilled_true(self):
        """
        """
        curriter = objective.Eval(None, None, 0.1 * self.tol, None)
        self.assertEqual(self.stopcrit.fulfilled(curriter, None), True)

    def test_fulfilled_false(self):
        """
        """
        curriter = objective.Eval(None, None, 10.0 * self.tol, None)
        self.assertEqual(self.stopcrit.fulfilled(curriter, None), False)

    def test_no_gradients(self):
        """
        """
        curriter = objective.Eval(1.0, 1.0, None, None)
        with self.assertRaises(AttributeError):
            self.stopcrit.fulfilled(curriter, None)

    def test_create_unfulfilled(self):
        """
        """
        unfulfilled_iter = self.stopcrit.create_unfulfilled(None)
        self.assertEqual(self.stopcrit.fulfilled(unfulfilled_iter, None),
                         False)


class TestDiffOfFctValues(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """
        self.tol = 1e-4
        self.stopcrit = stoppingcriterion.DiffOfFctValues(self.tol)

    def test_fulfilled_true(self):
        """
        """
        lastiter = objective.Eval(None, 123.4, None, None)
        curriter = objective.Eval(None, 123.4 + 0.1 * self.tol, None, None)
        self.assertEqual(self.stopcrit.fulfilled(curriter, lastiter), True)

    def test_fulfilled_false(self):
        """
        """
        lastiter = objective.Eval(None, 123.4, None, None)
        curriter = objective.Eval(None, 123.4 + 10.0 * self.tol, None, None)
        self.assertEqual(self.stopcrit.fulfilled(curriter, lastiter), False)

    def test_no_fvals(self):
        """
        """
        gooditer = objective.Eval(None, 1.0, None, None)
        baditer = objective.Eval(1.0, None, 1.0, 1.0)
        with self.assertRaises(AttributeError):
            self.stopcrit.fulfilled(gooditer, baditer)
        with self.assertRaises(AttributeError):
            self.stopcrit.fulfilled(baditer, gooditer)

    def test_create_unfulfilled(self):
        """
        """
        gooditer = objective.Eval(None, 1.0, None, None)
        unfulfilled_iter = self.stopcrit.create_unfulfilled(gooditer)
        self.assertEqual(self.stopcrit.fulfilled(gooditer, unfulfilled_iter),
                         False)
