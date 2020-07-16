"""
SteepestDescent.iterate() is tested
through TestGradientDescent, TestNewtonMethod, ...
"""

import unittest

import numpy as np

from probnum.optim import objective, linesearch, stoppingcriterion
from probnum.optim.deterministic import steepestdescent


class TestGradientDescent(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """

        def obj(x):
            """
            """
            return x.T @ x

        def grad(x):
            """
            """
            return 2 * x

        stopcrit = stoppingcriterion.NormOfGradient(1e-2)
        lsearch = linesearch.ConstantLearningRate(1e-1)
        self.gd = steepestdescent.GradientDescent(lsearch, stopcrit, maxit=10)
        self.objective = objective.Objective(obj, grad)
        self.curriter = self.objective.evaluate(np.array([11.234]))

    def test_compute_direction(self):
        """
        Checks whether compute_direction() is as expected for GD
        """
        direction = self.gd.compute_direction(self.curriter)
        self.assertEqual(direction, -2 * np.array([11.234]))

    def test_iterate(self):
        """
        Checks whether iterate() is as expected for GD
        """
        iteration = self.gd.iterate(self.curriter, self.objective)
        exp_iter = self.curriter.x - 1e-1 * self.curriter.dfx
        self.assertEqual(exp_iter, iteration.x)

    def test_no_grad_avail(self):
        """
        """
        mock = np.array([1.0])
        bad_iter = objective.Eval(mock, mock, None, mock)
        with self.assertRaises(ValueError):
            self.gd.compute_direction(bad_iter)


class TestNewtonMethod(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """

        def obj(x):
            """
            """
            return x.T @ x

        def grad(x):
            """
            """
            return 2 * x

        def hess(x):
            """
            """
            return 2 * np.ones((len(x), len(x)))

        stopcrit = stoppingcriterion.NormOfGradient(1e-2)
        lsearch = linesearch.ConstantLearningRate(1e-1)
        self.newton = steepestdescent.NewtonMethod(lsearch, stopcrit, maxit=10)
        self.objective = objective.Objective(obj, grad, hess)
        self.curriter = self.objective.evaluate(np.array([11.234]))

    def test_compute_direction(self):
        """
        """
        direction = self.newton.compute_direction(self.curriter)
        self.assertEqual(direction, -np.array([11.234]))

    def test_iterate(self):
        """
        """
        iteration = self.newton.iterate(self.curriter, self.objective)
        expdirection = -np.linalg.solve(self.curriter.ddfx, self.curriter.dfx)
        exp_iter = self.curriter.x + 1e-1 * expdirection
        self.assertEqual(exp_iter, iteration.x)

    def test_no_grad_avail(self):
        """
        """
        mock = np.array([1.0])
        bad_iter = objective.Eval(mock, mock, None, mock)
        with self.assertRaises(ValueError):
            self.newton.compute_direction(bad_iter)

    def test_no_hess_avail(self):
        """
        """
        mock = np.array([1.0])
        bad_iter = objective.Eval(mock, mock, mock, None)
        with self.assertRaises(ValueError):
            self.newton.compute_direction(bad_iter)


class TestLevenbergMarquardt(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """

        def obj(x):
            """
            """
            return x.T @ x

        def grad(x):
            """
            """
            return 2 * x

        def hess(x):
            """
            """
            return 2 * np.ones((len(x), len(x)))

        stopcrit = stoppingcriterion.NormOfGradient(1e-2)
        lsearch = linesearch.ConstantLearningRate(1e-1)
        self.dampingpar = 1e-05
        self.levmarq = steepestdescent.LevenbergMarquardt(self.dampingpar,
                                                          lsearch, stopcrit,
                                                          maxit=10)
        self.objective = objective.Objective(obj, grad, hess)
        self.curriter = self.objective.evaluate(np.array([11.234]))

    def test_negative_dampingpar(self):
        """
        """
        with self.assertRaises(ValueError):
            steepestdescent.LevenbergMarquardt(-1.0, None, None, None)

    def test_zero_dampingpar(self):
        """
        """
        with self.assertRaises(ValueError):
            steepestdescent.LevenbergMarquardt(0.0, None, None, None)

    def test_compute_direction(self):
        """
        """
        direction = self.levmarq.compute_direction(self.curriter)
        self.assertEqual(direction,
                         -2 / (2 + self.dampingpar) * np.array([11.234]))

    def test_iterate(self):
        """
        """
        iteration = self.levmarq.iterate(self.curriter, self.objective)
        expdirection = -np.linalg.solve(self.curriter.ddfx + self.dampingpar,
                                        self.curriter.dfx)
        exp_iter = self.curriter.x + 1e-1 * expdirection
        self.assertEqual(exp_iter, iteration.x)

    def test_no_grad_avail(self):
        """
        """
        mock = np.array([1.0])
        bad_iter = objective.Eval(mock, mock, None, mock)
        with self.assertRaises(ValueError):
            self.levmarq.compute_direction(bad_iter)

    def test_no_hess_avail(self):
        """
        """
        mock = np.array([1.0])
        bad_iter = objective.Eval(mock, mock, mock, None)
        with self.assertRaises(ValueError):
            self.levmarq.compute_direction(bad_iter)
