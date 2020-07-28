import unittest
import numpy as np

from probnum.prob import RandomVariable, Dirac
from probnum.prob.randomvariablelist import _RandomVariableList
from probnum.diffeq import probsolve_ivp
from probnum.diffeq.ode import lotkavolterra

from tests.testing import NumpyAssertions


class TestODESolution(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        initrv = RandomVariable(distribution=Dirac(20 * np.ones(2)))
        self.ivp = lotkavolterra([0.0, 0.5], initrv)
        step = 0.1
        self.solution = probsolve_ivp(self.ivp, step=step)

    def test_len(self):
        self.assertEqual(len(self.solution.t), len(self.solution))
        self.assertEqual(len(self.solution.y), len(self.solution))

    def test_t(self):
        self.assertArrayEqual(self.solution.t, np.sort(self.solution.t))

        self.assertApproxEqual(self.solution.t[0], self.ivp.t0)
        self.assertApproxEqual(self.solution.t[-1], self.ivp.tmax)

    def test_getitem(self):
        self.assertArrayEqual(self.solution[0].mean(), self.solution.y[0].mean())
        self.assertArrayEqual(self.solution[0].cov(), self.solution.y[0].cov())

        self.assertArrayEqual(self.solution[-1].mean(), self.solution.y[-1].mean())
        self.assertArrayEqual(self.solution[-1].cov(), self.solution.y[-1].cov())

        self.assertArrayEqual(self.solution[:].mean(), self.solution.y[:].mean())
        self.assertArrayEqual(self.solution[:].cov(), self.solution.y[:].cov())

    def test_y(self):
        self.assertEqual(type(self.solution.y), _RandomVariableList)

        self.assertEqual(len(self.solution.y[0].shape), 1)
        self.assertEqual(self.solution.y[0].shape[0], self.ivp.ndim)

    def test_dy(self):
        self.assertEqual(type(self.solution.dy), _RandomVariableList)

        self.assertEqual(len(self.solution.dy[0].shape), 1)
        self.assertEqual(self.solution.dy[0].shape[0], self.ivp.ndim)

    def test_call(self):
        # Results should coincide with the discrete solution for known t
        self.assertArrayEqual(self.solution(0.0).mean(), self.solution[0].mean())
        self.assertArrayEqual(self.solution(0.0).cov(), self.solution[0].cov())

        self.assertArrayEqual(self.solution(0.5).mean(), self.solution[-1].mean())
        self.assertArrayEqual(self.solution(0.5).cov(), self.solution[-1].cov())

        # t < t0 should raise an error
        with self.assertRaises(ValueError):
            self.solution(-0.5)

        # t0 < t < tmax
        self.solution(0.4127)

        # t > tmax
        self.solution(1.5)
