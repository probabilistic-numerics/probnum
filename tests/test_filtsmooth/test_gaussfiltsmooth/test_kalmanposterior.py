import numpy as np

from probnum.filtsmooth.gaussfiltsmooth import Kalman
from probnum._randomvariablelist import _RandomVariableList
from tests.testing import NumpyAssertions, chi_squared_statistic

from .filtsmooth_testcases import CarTrackingDDTestCase


class TestKalmanPosterior(CarTrackingDDTestCase, NumpyAssertions):
    def setUp(self):
        super().setup_cartracking()
        self.method = Kalman(self.dynmod, self.measmod, self.initrv)
        self.posterior = self.method.filter(self.obs, self.tms)

    def test_len(self):
        self.assertTrue(len(self.posterior) > 0)
        self.assertEqual(len(self.posterior.locations), len(self.posterior))
        self.assertEqual(len(self.posterior.state_rvs), len(self.posterior))

    def test_locations(self):
        self.assertArrayEqual(
            self.posterior.locations, np.sort(self.posterior.locations)
        )

        self.assertApproxEqual(self.posterior.locations[0], self.tms[0])
        self.assertApproxEqual(self.posterior.locations[-1], self.tms[-1])

    def test_getitem(self):
        self.assertArrayEqual(self.posterior[0].mean, self.posterior.state_rvs[0].mean)
        self.assertArrayEqual(self.posterior[0].cov, self.posterior.state_rvs[0].cov)

        self.assertArrayEqual(
            self.posterior[-1].mean, self.posterior.state_rvs[-1].mean
        )
        self.assertArrayEqual(self.posterior[-1].cov, self.posterior.state_rvs[-1].cov)

        self.assertArrayEqual(self.posterior[:].mean, self.posterior.state_rvs[:].mean)
        self.assertArrayEqual(self.posterior[:].cov, self.posterior.state_rvs[:].cov)

    def test_state_rvs(self):
        self.assertTrue(isinstance(self.posterior.state_rvs, _RandomVariableList))

        self.assertEqual(len(self.posterior.state_rvs[0].shape), 1)
        self.assertEqual(self.posterior.state_rvs[-1].shape, self.initrv.shape)

    def test_call_error_if_small(self):
        self.assertLess(-0.5, self.tms[0])
        with self.assertRaises(ValueError):
            self.posterior(-0.5)

    def test_call_vectorisation(self):
        locs = np.arange(0, 1, 20)
        evals = self.posterior(locs)
        self.assertEqual(len(evals), len(locs))

    def test_call_interpolation(self):
        self.assertLess(self.tms[0], 9.88)
        self.assertGreater(self.tms[-1], 9.88)
        self.assertTrue(9.88 not in self.tms)
        self.posterior(9.88)

    def test_call_to_discrete(self):
        self.assertEqual(self.tms[0], 0)
        self.assertArrayEqual(self.posterior(0.0).mean, self.posterior[0].mean)
        self.assertArrayEqual(self.posterior(0.0).cov, self.posterior[0].cov)

        self.assertEqual(self.tms[-1], 19.8)
        self.assertArrayEqual(self.posterior(19.8).mean, self.posterior[-1].mean)
        self.assertArrayEqual(self.posterior(19.8).cov, self.posterior[-1].cov)

        self.assertArrayEqual(self.posterior(self.tms[2]).mean, self.posterior[2].mean)
        self.assertArrayEqual(self.posterior(self.tms[5]).mean, self.posterior[5].mean)
        self.assertArrayEqual(
            self.posterior(self.tms[10]).mean, self.posterior[10].mean
        )

    def test_call_extrapolation(self):
        self.assertGreater(30, self.tms[-1])
        self.posterior(30)

    def test_sampling_all_locations_one_sample(self):
        sample = self.posterior.sample()

        with self.subTest(msg="Test output shape"):
            self.assertEqual(len(sample), len(self.posterior))

        with self.subTest(msg="Chi squared test"):
            chi_squared = chi_squared_statistic(
                sample, self.posterior[:].mean, self.posterior[:].cov
            )
            self.assertLess(chi_squared, 10.0)
            self.assertLess(0.1, chi_squared)

    def test_sampling_all_locations_multiple_samples(self):
        five_samples = self.posterior.sample(size=5)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(five_samples.shape[0], 5)
            self.assertEqual(five_samples.shape[1], len(self.posterior))

        with self.subTest(msg="Chi squared test"):
            chi_squared = np.array(
                [
                    chi_squared_statistic(
                        sample, self.posterior[:].mean, self.posterior[:].cov
                    )
                    for sample in five_samples
                ]
            ).mean()
            self.assertLess(chi_squared, 10.0)
            self.assertLess(0.1, chi_squared)

        with self.subTest(msg="non-integers rejected"):
            with self.assertRaises(NotImplementedError):
                self.posterior.sample(size=(2, 3))

    def test_sampling_two_locations_one_sample(self):
        locs = self.posterior.locations[[2, 3]]
        sample = self.posterior.sample(locations=locs)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(len(sample), 2)

        with self.subTest(msg="Chi squared test"):
            chi_squared = chi_squared_statistic(
                sample, self.posterior[:].mean[[2, 3]], self.posterior[:].cov[[2, 3]]
            )
            self.assertLess(chi_squared, 10.0)
            self.assertLess(0.1, chi_squared)

    def test_sampling_two_locations_multiple_samples(self):
        locs = self.posterior.locations[[2, 3]]
        five_samples = self.posterior.sample(locations=locs, size=5)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(five_samples.shape[0], 5)
            self.assertEqual(five_samples.shape[1], 2)

        with self.subTest(msg="Chi squared test"):
            chi_squared = np.array(
                [
                    chi_squared_statistic(
                        sample,
                        self.posterior[:].mean[[2, 3]],
                        self.posterior[:].cov[[2, 3]],
                    )
                    for sample in five_samples
                ]
            ).mean()
            self.assertLess(chi_squared, 10.0)
            self.assertLess(0.1, chi_squared)

        with self.subTest(msg="non-integers rejected"):
            with self.assertRaises(NotImplementedError):
                self.posterior.sample(locations=locs, size=(2, 3))

    def test_sampling_many_locations_one_sample(self):
        locs = np.arange(0.0, 0.5, 0.025)
        sample = self.posterior.sample(locations=locs)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(len(sample), len(locs))

        with self.subTest(msg="Chi squared test"):
            chi_squared = chi_squared_statistic(
                sample, self.posterior(locs).mean, self.posterior(locs).cov
            )
            self.assertLess(chi_squared, 10.0)
            self.assertLess(0.1, chi_squared)

    def test_sampling_many_locations_multiple_samples(self):
        locs = np.arange(0.0, 0.5, 0.025)
        five_samples = self.posterior.sample(locations=locs, size=5)

        with self.subTest(msg="Test output shape"):
            self.assertEqual(five_samples.shape[0], 5)
            self.assertEqual(five_samples.shape[1], len(locs))

        with self.subTest(msg="Chi squared test"):
            chi_squared = np.array(
                [
                    chi_squared_statistic(
                        sample, self.posterior(locs).mean, self.posterior(locs).cov
                    )
                    for sample in five_samples
                ]
            ).mean()
            self.assertLess(chi_squared, 10.0)
            self.assertLess(0.1, chi_squared)

        with self.subTest(msg="non-integers rejected"):
            with self.assertRaises(NotImplementedError):
                self.posterior.sample(locations=locs, size=(2, 3))
