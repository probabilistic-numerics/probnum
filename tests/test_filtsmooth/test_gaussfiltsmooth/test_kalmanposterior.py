import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.filtsmooth.statespace as pnfss
import probnum.random_variables as pnrv
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


#
# @pytest.fixture
# def posterior(kalman, problem):
#     *_, obs, times, states = problem
#     return kalman.filtsmooth(obs, times)
#


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


def test_state_rvs(posterior):
    assert isinstance(posterior.state_rvs, _RandomVariableList)
    assert len(posterior.state_rvs[0].shape) == 1


def test_call_error_if_small(posterior):
    assert -0.5 < posterior.locations[0]
    with pytest.raises(ValueError):
        posterior(-0.5)


def test_call_vectorisation(posterior):
    locs = np.arange(0, 1, 20)
    evals = posterior(locs)
    assert len(evals) == len(locs)


def test_call_interpolation(posterior):
    assert posterior.locations[0] < 9.88 < posterior.locations[-1]
    assert 9.88 not in posterior.locations
    out_rv = posterior(9.88)
    assert isinstance(out_rv, pnrv.Normal)


def test_call_to_discrete(posterior):
    """Called at a grid point, the respective disrete solution is returned."""

    first_point = posterior.locations[0]
    np.testing.assert_allclose(posterior(first_point).mean, posterior[0].mean)
    np.testing.assert_allclose(posterior(first_point).cov, posterior[0].cov)

    final_point = posterior.locations[-1]
    np.testing.assert_allclose(posterior(final_point).mean, posterior[-1].mean)
    np.testing.assert_allclose(posterior(final_point).cov, posterior[-1].cov)

    mid_point = posterior.locations[4]
    np.testing.assert_allclose(posterior(mid_point).mean, posterior[4].mean)
    np.testing.assert_allclose(posterior(mid_point).cov, posterior[4].cov)


def test_call_extrapolation(posterior):
    assert posterior.locations[-1] < 30.0
    out_rv = posterior(30.0)
    assert isinstance(out_rv, pnrv.Normal)


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
