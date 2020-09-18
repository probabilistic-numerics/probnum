import unittest
import numpy as np

from probnum.random_variables import Dirac
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

    def test_sampling_all_locations_one_sample(self):
        sample = self.solution.sample()

        with self.subTest(msg="Test output shape"):
            self.assertEqual(len(sample), len(self.solution))

        with self.subTest(msg="Chi squared test"):
            chi_squared = chi_squared_statistic(
                sample[1:], self.solution[1:].mean, self.solution[1:].cov
            )
            # extreme values bc. higher order priors are poorly calibrated
            self.assertLess(chi_squared, 200.0)
            self.assertLess(0.005, chi_squared)

    def test_sampling_all_locations_multiple_samples(self):
        five_samples = self.solution.sample(size=5)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(five_samples.shape[0], 5)
            self.assertEqual(five_samples.shape[1], len(self.solution))

        with self.subTest(msg="Chi squared test"):
            chi_squared = np.array(
                [
                    chi_squared_statistic(
                        sample[1:], self.solution[1:].mean, self.solution[1:].cov
                    )
                    for sample in five_samples
                ]
            ).mean()

            # extreme values bc. higher order priors are poorly calibrated
            self.assertLess(chi_squared, 200.0)
            self.assertLess(0.005, chi_squared)

        with self.subTest(msg="non-integers rejected"):
            with self.assertRaises(NotImplementedError):
                self.solution.sample(size=(2, 3))

    def test_sampling_two_locations_one_sample(self):
        locs = self.solution.t[[2, 3]]
        sample = self.solution.sample(t=locs)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(len(sample), 2)

        with self.subTest(msg="Chi squared test"):
            chi_squared = chi_squared_statistic(
                sample, self.solution[:].mean[[2, 3]], self.solution[:].cov[[2, 3]]
            )
            self.assertLess(chi_squared, 200.0)
            self.assertLess(0.005, chi_squared)

    def test_sampling_two_locations_multiple_samples(self):
        locs = self.solution.t[[2, 3]]
        five_samples = self.solution.sample(t=locs, size=5)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(five_samples.shape[0], 5)
            self.assertEqual(five_samples.shape[1], 2)

        with self.subTest(msg="Chi squared test"):
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
            self.assertLess(chi_squared, 200.0)
            self.assertLess(0.005, chi_squared)

        with self.subTest(msg="non-integers rejected"):
            with self.assertRaises(NotImplementedError):
                self.solution.sample(t=locs, size=(2, 3))

    def test_sampling_many_locations_one_sample(self):
        locs = np.arange(0.1, 0.5, 0.025)
        sample = self.solution.sample(locs)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(len(sample), len(locs))

        with self.subTest(msg="Chi squared test"):
            chi_squared = chi_squared_statistic(
                sample, self.solution(locs).mean, self.solution(locs).cov
            )
            self.assertLess(chi_squared, 200.0)
            self.assertLess(0.005, chi_squared)

    def test_sampling_many_locations_multiple_samples(self):
        locs = np.arange(0.0, 0.5, 0.025)
        five_samples = self.solution.sample(t=locs, size=5)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(five_samples.shape[0], 5)
            self.assertEqual(five_samples.shape[1], len(locs))

        with self.subTest(msg="Chi squared test"):
            chi_squared = np.array(
                [
                    chi_squared_statistic(
                        sample[1:],
                        self.solution(locs)[1:].mean,
                        self.solution(locs)[1:].cov,
                    )
                    for sample in five_samples
                ]
            ).mean()
            self.assertLess(chi_squared, 5000.0)  # is this still a test???
            self.assertLess(0.005, chi_squared)

        with self.subTest(msg="non-integers rejected"):
            with self.assertRaises(NotImplementedError):
                self.solution.sample(t=locs, size=(2, 3))


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
