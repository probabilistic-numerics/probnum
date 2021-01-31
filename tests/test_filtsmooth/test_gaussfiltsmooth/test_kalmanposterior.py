import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.filtsmooth.statespace as pnfss
from probnum._randomvariablelist import _RandomVariableList
from probnum.filtsmooth.gaussfiltsmooth import Kalman
from tests.testing import NumpyAssertions, chi_squared_statistic

from .filtsmooth_testcases import CarTrackingDDTestCase, car_tracking


@pytest.fixture
def problem():
    """Car-tracking problem."""
    problem = car_tracking()
    dynmod, measmod, initrv, info = problem

    times = np.arange(0, info["tmax"], info["dt"])
    states, obs = pnfss.generate(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    return dynmod, measmod, initrv, info, obs, times, states


@pytest.fixture
def kalman(problem):
    dynmod, measmod, initrv, *_ = problem
    return pnfs.Kalman(dynmod, measmod, initrv)


@pytest.fixture
def posterior(kalman, problem):
    *_, obs, times, states = problem
    return kalman.filter(obs, times)


@pytest.fixture
def posterior(kalman, problem):
    *_, obs, times, states = problem
    return kalman.filtsmooth(obs, times)


def test_len(posterior):
    assert len(posterior) > 0
    assert len(posterior.locations) == len(posterior)
    assert len(posterior.state_rvs) == len(posterior)


def test_locations(posterior, problem):
    *_, obs, times, states = problem
    np.testing.assert_allclose(posterior.locations, np.sort(posterior.locations))
    np.testing.assert_allclose(posterior.locations, times)


def test_getitem(posterior):
    np.testing.assert_allclose(posterior[0].mean, posterior.state_rvs[0].mean)
    np.testing.assert_allclose(posterior[0].cov, posterior.state_rvs[0].cov)

    np.testing.assert_allclose(posterior[-1].mean, posterior.state_rvs[-1].mean)
    np.testing.assert_allclose(posterior[-1].cov, posterior.state_rvs[-1].cov)

    np.testing.assert_allclose(posterior[:].mean, posterior.state_rvs[:].mean)
    np.testing.assert_allclose(posterior[:].cov, posterior.state_rvs[:].cov)


class TestKalmanPosterior(CarTrackingDDTestCase, NumpyAssertions):
    def setUp(self):
        super().setup_cartracking()
        self.method = Kalman(self.dynmod, self.measmod, self.initrv)
        self.posterior = self.method.filter(self.obs, self.tms)

    #
    # def test_len(self):
    #     self.assertTrue(len(self.posterior) > 0)
    #     self.assertEqual(len(self.posterior.locations), len(self.posterior))
    #     self.assertEqual(len(self.posterior.state_rvs), len(self.posterior))
    #
    # def test_locations(self):
    #     self.assertArrayEqual(
    #         self.posterior.locations, np.sort(self.posterior.locations)
    #     )
    #
    #     self.assertApproxEqual(self.posterior.locations[0], self.tms[0])
    #     self.assertApproxEqual(self.posterior.locations[-1], self.tms[-1])
    #
    # def test_getitem(self):
    #     self.assertArrayEqual(self.posterior[0].mean, self.posterior.state_rvs[0].mean)
    #     self.assertArrayEqual(self.posterior[0].cov, self.posterior.state_rvs[0].cov)
    #
    #     self.assertArrayEqual(
    #         self.posterior[-1].mean, self.posterior.state_rvs[-1].mean
    #     )
    #     self.assertArrayEqual(self.posterior[-1].cov, self.posterior.state_rvs[-1].cov)
    #
    #     self.assertArrayEqual(self.posterior[:].mean, self.posterior.state_rvs[:].mean)
    #     self.assertArrayEqual(self.posterior[:].cov, self.posterior.state_rvs[:].cov)

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


class TestKalmanPosteriorSampling(CarTrackingDDTestCase, NumpyAssertions):
    # There is no check as to whether the samples make sense...
    def setUp(self):
        super().setup_cartracking()
        self.method = Kalman(self.dynmod, self.measmod, self.initrv)
        self.posterior = self.method.filter(self.obs, self.tms)

    def test_output_shape(self):
        loc_inputs = [
            None,
            self.posterior.locations[[2, 3]],
            np.arange(0.0, 0.5, 0.025),
        ]
        dim = (self.method.dynamics_model.dimension,)
        single_sample_shapes = [
            (len(self.posterior), self.method.dynamics_model.dimension),
            (2, self.method.dynamics_model.dimension),
            (len(loc_inputs[-1]), self.method.dynamics_model.dimension),
        ]

        for size in [(), (5,), (2, 3, 4)]:
            for loc, loc_shape in zip(loc_inputs, single_sample_shapes):
                with self.subTest(size=size, loc=loc):
                    sample = self.posterior.sample(locations=loc, size=size)
                    if size == ():
                        self.assertEqual(sample.shape, loc_shape)
                    else:
                        self.assertEqual(sample.shape, size + loc_shape)

    def test_sampling_all_locations_multiple_samples(self):
        five_samples = self.posterior.sample(size=5)

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

    def test_sampling_two_locations_multiple_samples(self):
        locs = self.posterior.locations[[2, 3]]
        five_samples = self.posterior.sample(locations=locs, size=5)

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

    def test_sampling_many_locations_multiple_samples(self):
        locs = np.arange(0.0, 0.5, 0.025)
        five_samples = self.posterior.sample(locations=locs, size=5)

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
