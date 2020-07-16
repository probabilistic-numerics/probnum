"""
"""

import unittest

import numpy as np

from probnum.optim import objective, linesearch, stoppingcriterion
from probnum.optim.deterministic import randomsearch


class TestRandomSearch(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """

        def obj(x):
            """
            """
            return x.T @ x

        stopcrit = stoppingcriterion.DiffOfFctValues(0.0)
        lsearch = linesearch.ConstantLearningRate(1e-1)
        self.rs = randomsearch.RandomSearch(lsearch, stopcrit, maxit=100)
        self.objective = objective.Objective(obj)
        self.initval = np.array([11.234])

    def test_nonconstant_lrate(self):
        """
        """
        stopcrit = stoppingcriterion.DiffOfFctValues(1e-1)
        lsearch = linesearch.BacktrackingLineSearch()
        with self.assertRaises(AttributeError):
            randomsearch.RandomSearch(lsearch, stopcrit, maxit=10)

    def test_iterate(self):
        """
        Checks for 30 iterations, whether each function value is
        less or equal than the previous one, and if it is smaller,
        if the state is closer to the true minimum than the previous.
        """
        last = self.objective.evaluate(self.initval)
        curr = self.rs.iterate(last, self.objective)
        for idx in range(30):
            self.assertLessEqual(curr.fx, last.fx)
            if curr.fx < last.fx:
                self.assertLess(np.linalg.norm(curr.x),
                                np.linalg.norm(last.x))

    def test_no_state(self):
        """
        """
        mock = np.array([1234.5678])
        mockiter = objective.Eval(None, mock, mock, mock)
        with self.assertRaises(ValueError):
            self.rs.iterate(mockiter, self.objective)

    def test_no_fct_evaluation(self):
        """
        """
        mock = np.array([1234.5678])
        mockiter = objective.Eval(mock, None, mock, mock)
        with self.assertRaises(ValueError):
            self.rs.iterate(mockiter, self.objective)
