import numpy as np

from probnum.filtsmooth.gaussfiltsmooth import Kalman
from probnum._randomvariablelist import _RandomVariableList
from tests.testing import NumpyAssertions

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

        self.assertArrayEqual(
            self.posterior[:].mean(), self.posterior.state_rvs[:].mean()
        )
        self.assertArrayEqual(
            self.posterior[:].cov(), self.posterior.state_rvs[:].cov()
        )

    def test_state_rvs(self):
        self.assertTrue(isinstance(self.posterior.state_rvs, _RandomVariableList))

        self.assertEqual(len(self.posterior.state_rvs[0].shape), 1)
        self.assertEqual(self.posterior.state_rvs[-1].shape, self.initrv.shape)

    def test_call_error_if_small(self):
        self.assertLess(-0.5, self.tms[0])
        with self.assertRaises(ValueError):
            self.posterior(-0.5)

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
        self.posterior(30, smoothed=False)
