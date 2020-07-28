import numpy as np

from probnum.prob.randomvariablelist import _RandomVariableList
from probnum.filtsmooth.gaussfiltsmooth import Kalman

from .filtsmooth_testcases import CarTrackingDDTestCase
from tests.testing import NumpyAssertions


class TestKalmanPosterior(CarTrackingDDTestCase, NumpyAssertions):
    def setUp(self):
        super().setup_cartracking()
        self.method = Kalman(self.dynmod, self.measmod, self.initrv)
        self.posterior = self.method.filter(self.obs, self.tms)

    def test_len(self):
        self.assertEqual(len(self.posterior.locations), len(self.posterior))
        self.assertEqual(len(self.posterior.state_rvs), len(self.posterior))

    def test_locations(self):
        self.assertArrayEqual(
            self.posterior.locations, np.sort(self.posterior.locations)
        )

        self.assertApproxEqual(self.posterior.locations[0], self.tms[0])
        self.assertApproxEqual(self.posterior.locations[-1], self.tms[-1])

    def test_getitem(self):
        self.assertArrayEqual(
            self.posterior[0].mean(), self.posterior.state_rvs[0].mean()
        )
        self.assertArrayEqual(
            self.posterior[0].cov(), self.posterior.state_rvs[0].cov()
        )

        self.assertArrayEqual(
            self.posterior[-1].mean(), self.posterior.state_rvs[-1].mean()
        )
        self.assertArrayEqual(
            self.posterior[-1].cov(), self.posterior.state_rvs[-1].cov()
        )

        self.assertArrayEqual(
            self.posterior[:].mean(), self.posterior.state_rvs[:].mean()
        )
        self.assertArrayEqual(
            self.posterior[:].cov(), self.posterior.state_rvs[:].cov()
        )

    def test_state_rvs(self):
        self.assertEqual(type(self.posterior.state_rvs), _RandomVariableList)

        self.assertEqual(len(self.posterior.state_rvs[0].shape), 1)
        self.assertEqual(self.posterior.state_rvs[-1].shape, self.initrv.shape)

    def test_call(self):
        # Results should coincide with the discrete posterior for known t
        self.assertEqual(self.tms[0], 0)
        self.assertArrayEqual(self.posterior(0.0).mean(), self.posterior[0].mean())
        self.assertArrayEqual(self.posterior(0.0).cov(), self.posterior[0].cov())

        self.assertEqual(self.tms[-1], 19.8)
        self.assertArrayEqual(self.posterior(19.8).mean(), self.posterior[-1].mean())
        self.assertArrayEqual(self.posterior(19.8).cov(), self.posterior[-1].cov())

        # t < t0 should raise an error
        self.assertLess(-0.5, self.tms[0])
        with self.assertRaises(ValueError):
            self.posterior(-0.5)

        # t0 < t < tmax
        self.assertLess(self.tms[0], 9.88)
        self.assertGreater(self.tms[-1], 9.88)
        self.assertTrue(9.88 not in self.tms)
        self.posterior(9.88)

        # t > tmax
        self.assertGreater(30, self.tms[-1])
        self.posterior(30)
