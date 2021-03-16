import unittest

import numpy as np

import probnum.filtsmooth as pnfs
from probnum._randomvariablelist import _RandomVariableList
from probnum.diffeq import KalmanODESolution, probsolve_ivp
from probnum.diffeq.ode import logistic, lotkavolterra
from probnum.randvars import Constant, Normal
from tests.testing import NumpyAssertions


class TestODESolution(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        initrv = Constant(20 * np.ones(2))
        self.ivp = lotkavolterra([0.0, 0.5], initrv)
        step = 0.1
        f = self.ivp.rhs
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean
        self.solution = probsolve_ivp(f, t0, tmax, y0, step=step, adaptive=False)

    def test_len(self):
        self.assertTrue(len(self.solution) > 0)
        self.assertEqual(len(self.solution.t), len(self.solution))
        self.assertEqual(len(self.solution.y), len(self.solution))

    def test_t(self):
        self.assertArrayEqual(self.solution.t, np.sort(self.solution.t))

        self.assertApproxEqual(self.solution.t[0], self.ivp.t0)
        self.assertApproxEqual(self.solution.t[-1], self.ivp.tmax)

    def test_getitem(self):
        self.assertArrayEqual(self.solution[0].mean, self.solution.y[0].mean)
        self.assertArrayEqual(self.solution[0].cov, self.solution.y[0].cov)

        self.assertArrayEqual(self.solution[-1].mean, self.solution.y[-1].mean)
        self.assertArrayEqual(self.solution[-1].cov, self.solution.y[-1].cov)

        self.assertArrayEqual(self.solution[:].mean, self.solution.y[:].mean)
        self.assertArrayEqual(self.solution[:].cov, self.solution.y[:].cov)

    def test_y(self):
        self.assertTrue(isinstance(self.solution.y, _RandomVariableList))

        self.assertEqual(len(self.solution.y[0].shape), 1)
        self.assertEqual(self.solution.y[0].shape[0], self.ivp.dimension)

    def test_dy(self):
        self.assertTrue(isinstance(self.solution.dy, _RandomVariableList))

        self.assertEqual(len(self.solution.dy[0].shape), 1)
        self.assertEqual(self.solution.dy[0].shape[0], self.ivp.dimension)

    def test_call_error_if_small(self):
        t = self.ivp.t0 - 0.5
        self.assertLess(t, self.solution.t[0])
        with self.assertRaises(ValueError):
            self.solution(t)

    def test_call_interpolation(self):
        t = self.ivp.t0 + (self.ivp.tmax - self.ivp.t0) / np.pi
        self.assertLess(self.solution.t[0], t)
        self.assertGreater(self.solution.t[-1], t)
        self.assertTrue(t not in self.solution.t)
        self.solution(t)

    def test_call_correctness(self):
        t = self.ivp.t0 + 1e-6
        self.assertAllClose(
            self.solution[0].mean, self.solution(t).mean, atol=1e-4, rtol=0
        )

    def test_call_endpoints(self):
        self.assertArrayEqual(self.solution(self.ivp.t0).mean, self.solution[0].mean)
        self.assertArrayEqual(self.solution(self.ivp.t0).cov, self.solution[0].cov)

        self.assertArrayEqual(self.solution(self.ivp.tmax).mean, self.solution[-1].mean)
        self.assertArrayEqual(self.solution(self.ivp.tmax).cov, self.solution[-1].cov)

    def test_call_extrapolation(self):
        t = 1.1 * self.ivp.tmax
        self.assertGreater(t, self.solution.t[-1])
        self.solution(t)

    def test_filtering_solution(self):
        filtpost = self.solution.filtering_solution
        self.assertIsInstance(filtpost, KalmanODESolution)
        self.assertIsInstance(filtpost.kalman_posterior, pnfs.FilteringPosterior)


class TestODESolutionHigherOrderPrior(TestODESolution):
    """Same as above, but higher-order prior to test for a different dimensionality."""

    def setUp(self):
        initrv = Constant(20 * np.ones(2))
        self.ivp = lotkavolterra([0.0, 0.5], initrv)
        step = 0.1
        f = self.ivp.rhs
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean

        self.solution = probsolve_ivp(
            f, t0, tmax, y0, algo_order=3, step=step, adaptive=False
        )


class TestODESolutionOneDimODE(TestODESolution):
    """Same as above, but 1d IVP to test for a different dimensionality."""

    def setUp(self):
        initrv = Constant(0.1 * np.ones(1))
        self.ivp = logistic([0.0, 1.5], initrv)
        step = 0.1
        f = self.ivp.rhs
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean

        self.solution = probsolve_ivp(
            f, t0, tmax, y0, algo_order=3, step=step, adaptive=False
        )


class TestODESolutionAdaptive(TestODESolution):
    """Same as above, but adaptive steps."""

    def setUp(self):
        super().setUp()
        f = self.ivp.rhs
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean

        self.solution = probsolve_ivp(f, t0, tmax, y0, algo_order=2, atol=0.1, rtol=0.1)


class TestODESolutionSampling(unittest.TestCase):
    def setUp(self):
        initrv = Normal(
            20 * np.ones(2), 0.1 * np.eye(2), cov_cholesky=np.sqrt(0.1) * np.eye(2)
        )
        self.ivp = lotkavolterra([0.0, 0.5], initrv)
        step = 0.1
        f = self.ivp.rhs
        t0, tmax = self.ivp.timespan
        y0 = self.ivp.initrv.mean

        self.solution = probsolve_ivp(f, t0, tmax, y0, step=step, adaptive=False)

    def test_output_shape(self):
        loc_inputs = [
            None,
            self.solution.t[[2, 3]],
            np.arange(0.0, 0.5, 0.05),
        ]
        single_sample_shapes = [
            (len(self.solution), self.ivp.dimension),
            (2, self.ivp.dimension),
            (len(loc_inputs[-1]), self.ivp.dimension),
        ]

        for size in [(), (2,), (2, 3, 4)]:
            for loc, loc_shape in zip(loc_inputs, single_sample_shapes):
                with self.subTest(size=size, loc=loc):
                    sample = self.solution.sample(t=loc, size=size)
                    if size == ():
                        self.assertEqual(sample.shape, loc_shape)
                    else:
                        self.assertEqual(sample.shape, size + loc_shape)
