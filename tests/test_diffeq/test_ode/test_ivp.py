import unittest

import numpy as np

from probnum.diffeq.ode import ivp
from probnum.randvars import Constant

TEST_NDIM = 3


class TestIVP(unittest.TestCase):
    def setUp(self):
        def rhs_(t, x):
            return -x

        def jac_(t, x):
            return -np.eye(len(x))

        def sol_(t):
            return np.exp(-t) * np.ones(TEST_NDIM)

        some_center = np.random.rand(TEST_NDIM)
        rv = Constant(some_center)
        self.mockivp = ivp.IVP(
            (0.0, np.random.rand()), rv, rhs=rhs_, jac=jac_, sol=sol_
        )

    def test_rhs(self):
        some_x = np.random.rand(TEST_NDIM)
        some_t = np.random.rand()
        out = self.mockivp.rhs(some_t, some_x)
        self.assertEqual(len(out), TEST_NDIM)

    def test_jacobian(self):
        some_x = np.random.rand(TEST_NDIM)
        some_t = np.random.rand()
        out = self.mockivp.jacobian(some_t, some_x)
        self.assertEqual(out.shape[0], TEST_NDIM)
        self.assertEqual(out.shape[1], TEST_NDIM)

    def test_solution(self):
        some_t = np.random.rand()
        out = self.mockivp.solution(some_t)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape[0], TEST_NDIM)

    def test_initialdistribution(self):
        __ = self.mockivp.initialdistribution

    def test_timespan(self):
        __, __ = self.mockivp.timespan
