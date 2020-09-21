import unittest
import numpy as np

from probnum.random_variables import Dirac, Normal
from probnum._randomvariablelist import _RandomVariableList
from probnum.diffeq import probsolve_ivp
from probnum.diffeq.ode import lotkavolterra, logistic

from tests.testing import NumpyAssertions, chi_squared_statistic


class TestODESolution(unittest.TestCase, NumpyAssertions):
    def setUp(self):
        initrv = Dirac(20 * np.ones(2))
        self.ivp = lotkavolterra([0.0, 0.5], initrv)
        step = 0.1
        self.solution = probsolve_ivp(self.ivp, step=step)

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
        self.assertEqual(self.solution.y[0].shape[0], self.ivp.ndim)

    def test_dy(self):
        self.assertTrue(isinstance(self.solution.dy, _RandomVariableList))

        self.assertEqual(len(self.solution.dy[0].shape), 1)
        self.assertEqual(self.solution.dy[0].shape[0], self.ivp.ndim)

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


class TestODESolutionHigherOrderPrior(TestODESolution):
    """Same as above, but higher-order prior to test for a different dimensionality"""

    def setUp(self):
        initrv = Dirac(20 * np.ones(2))
        self.ivp = lotkavolterra([0.0, 0.5], initrv)
        step = 0.1
        self.solution = probsolve_ivp(self.ivp, which_prior="ibm3", step=step)


class TestODESolutionOneDimODE(TestODESolution):
    """Same as above, but 1d IVP to test for a different dimensionality"""

    def setUp(self):
        initrv = Dirac(0.1 * np.ones(1))
        self.ivp = logistic([0.0, 1.5], initrv)
        step = 0.1
        self.solution = probsolve_ivp(self.ivp, which_prior="ibm3", step=step)


class TestODESolutionAdaptive(TestODESolution):
    """Same as above, but adaptive steps"""

    def setUp(self):
        super().setUp()
        self.solution = probsolve_ivp(self.ivp, which_prior="ibm2", tol=0.1)


class TestODESolutionSampling(unittest.TestCase):
    def setUp(self):
        initrv = Normal(20 * np.ones(2), 0.1 * np.eye(2))
        self.ivp = lotkavolterra([0.0, 0.5], initrv)
        step = 0.1
        self.solution = probsolve_ivp(self.ivp, step=step)

    def test_output_shape(self):
        loc_inputs = [
            None,
            self.solution.t[[2, 3]],
            np.arange(0.0, 0.5, 0.025),
        ]
        single_sample_shapes = [
            (len(self.solution), self.ivp.ndim),
            (2, self.ivp.ndim),
            (len(loc_inputs[-1]), self.ivp.ndim),
        ]

        for size in [(), (5,), (2, 3, 4)]:
            for loc, loc_shape in zip(loc_inputs, single_sample_shapes):
                with self.subTest(size=size, loc=loc):
                    sample = self.solution.sample(t=loc, size=size)
                    if size == ():
                        self.assertEqual(sample.shape, loc_shape)
                    else:
                        self.assertEqual(sample.shape, size + loc_shape)

    def test_sampling_all_locations_multiple_samples(self):
        five_samples = self.solution.sample(size=5)

        chi_squared = np.array(
            [
                chi_squared_statistic(
                    sample, self.solution[:].mean, self.solution[:].cov
                )
                for sample in five_samples
            ]
        ).mean()
        self.assertLess(chi_squared, 10.0)
        self.assertLess(0.1, chi_squared)

    def test_sampling_two_locations_multiple_samples(self):
        locs = self.solution.t[[2, 3]]
        five_samples = self.solution.sample(t=locs, size=5)

        chi_squared = np.array(
            [
                chi_squared_statistic(
                    sample,
                    self.solution[:].mean[[2, 3]],
                    self.solution[:].cov[[2, 3]],
                )
                for sample in five_samples
            ]
        ).mean()
        self.assertLess(chi_squared, 10.0)
        self.assertLess(0.1, chi_squared)

    def test_sampling_many_locations_multiple_samples(self):
        locs = np.arange(0.0, 0.5, 0.025)
        five_samples = self.solution.sample(t=locs, size=5)

        chi_squared = np.array(
            [
                chi_squared_statistic(
                    sample, self.solution(locs).mean, self.solution(locs).cov
                )
                for sample in five_samples
            ]
        ).mean()
        self.assertLess(chi_squared, 10.0)
        self.assertLess(0.1, chi_squared)
