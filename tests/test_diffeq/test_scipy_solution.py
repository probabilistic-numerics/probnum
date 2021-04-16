# -*- coding: utf-8 -*-
"""Created on Wed Jan 13 11:47:38 2021.

@author: Nina Effenberger
"""

import unittest

import numpy as np
import scipy.integrate as sci
from pn_ode_benchmarks import scipy_solver

import probnum.diffeq as pnd
import probnum.random_variables as pnrv
from probnum.diffeq import scipysolution


class TestScipyODESolution:
    def test_t(self):
        """ts in scipy is t in probnum wrapper/ScipyODESolution."""
        scipy_t = self.scipy_solution.ts
        probnum_t = self.solution.t
        self.assertAlmostEqual(scipy_t.all(), probnum_t.all())

    def test_y(self):
        """states in probnum are RVs."""
        scipy_states = np.array(self.scipy_solution(self.scipy_solution.ts)).T
        probnum_states = np.array(self.solution.y.mean)
        self.assertAlmostEqual(scipy_states.all(), probnum_states.all())

    def test_call__(self):
        scipy_call = self.scipy_solution(self.scipy_solution.ts)
        probnum_call = self.solution(self.scipy_solution.ts).mean
        probnum_reshaped = probnum_call.reshape(scipy_call.shape)
        self.assertAlmostEqual(scipy_call.all(), probnum_reshaped.all())

    def test_len__(self):
        scipy_len = len(self.scipy_solution.ts)
        probnum_len = len(self.solution)
        self.assertAlmostEqual(scipy_len, probnum_len)

    def test_getitem__(self):
        scipy_item = self.scipy_solution.interpolants[1](self.scipy_solution.ts[1])
        probnum_item = self.solution[1]
        self.assertAlmostEqual(scipy_item, probnum_item)

    def test_sample(self):
        probnum_sample = self.solution.sample(5)
        self.assertEqual(probnum_sample, "Not possible")


class TestRungeKutta45(TestScipyODESolution, unittest.TestCase):
    """test class for RK23."""

    def setUp(self):
        initrv = pnrv.Constant(np.array([0.1]))
        self.ivp = pnd.logistic([0.0, 10], initrv)
        steprule = pnd.ConstantSteps(0.1)
        testsolver = sci.RK45(
            self.ivp.rhs, self.ivp.t0, self.ivp.initrv.mean, self.ivp.tmax
        )
        self.solver = scipy_solver.ScipyRungeKutta(testsolver, order=4)
        self.solution = self.solver.solve(steprule)
        self.scipy_solution = self.solution.scipy_solution
