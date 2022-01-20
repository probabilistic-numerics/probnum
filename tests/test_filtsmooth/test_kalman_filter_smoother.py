"""Test for the convenience functions."""

import numpy as np
import pytest

from probnum import filtsmooth


@pytest.fixture(name="prior_dimension")
def fixture_prior_dimension():
    return 3


@pytest.fixture(name="measurement_dimension")
def fixture_measurement_dimension():
    return 2


@pytest.fixture(name="number_of_measurements")
def fixture_number_of_measurements():
    return 10


@pytest.fixture(name="observations")
def fixture_observations(measurement_dimension, number_of_measurements):
    obs = np.arange(measurement_dimension * number_of_measurements)
    return obs.reshape((number_of_measurements, measurement_dimension))


@pytest.fixture(name="locations")
def fixture_locations(number_of_measurements):
    return np.arange(number_of_measurements)


@pytest.fixture(name="F")
def fixture_F(prior_dimension):
    return np.eye(prior_dimension)


@pytest.fixture(name="L")
def fixture_L(prior_dimension):
    return np.eye(prior_dimension)


@pytest.fixture(name="H")
def fixture_H(measurement_dimension, prior_dimension):
    return np.eye(measurement_dimension, prior_dimension)


@pytest.fixture(name="R")
def fixture_R(measurement_dimension):
    return np.eye(measurement_dimension)


@pytest.fixture(name="m0")
def fixture_m0(prior_dimension):
    return np.arange(prior_dimension)


@pytest.fixture(name="C0")
def fixture_C0(prior_dimension):
    return np.eye(prior_dimension)


@pytest.mark.parametrize("prior_model", ["continuous", "discrete"])
def test_kalman_filter(observations, locations, F, L, H, R, m0, C0, prior_model):
    posterior = filtsmooth.filter_kalman(
        observations=observations,
        locations=locations,
        F=F,
        L=L,
        H=H,
        R=R,
        m0=m0,
        C0=C0,
        prior_model=prior_model,
    )
    assert isinstance(posterior, filtsmooth.gaussian.FilteringPosterior)


@pytest.mark.parametrize("prior_model", ["continuous", "discrete"])
def test_rauch_tung_striebel_smoother(
    observations, locations, F, L, H, R, m0, C0, prior_model
):
    posterior = filtsmooth.smooth_rts(
        observations=observations,
        locations=locations,
        F=F,
        L=L,
        H=H,
        R=R,
        m0=m0,
        C0=C0,
        prior_model=prior_model,
    )
    assert isinstance(posterior, filtsmooth.gaussian.SmoothingPosterior)


def test_kalman_filter_raise_error(observations, locations, F, L, H, R, m0, C0):
    """Neither continuous nor discrete prior model raises a value error."""
    with pytest.raises(ValueError):
        filtsmooth.filter_kalman(
            observations=observations,
            locations=locations,
            F=F,
            L=L,
            H=H,
            R=R,
            m0=m0,
            C0=C0,
            prior_model="neither_continuous_nor_discrete",
        )


def test_rauch_tung_striebel_smoother_raise_error(
    observations, locations, F, L, H, R, m0, C0
):
    """Neither continuous nor discrete prior model raises a value error."""
    with pytest.raises(ValueError):
        filtsmooth.smooth_rts(
            observations=observations,
            locations=locations,
            F=F,
            L=L,
            H=H,
            R=R,
            m0=m0,
            C0=C0,
            prior_model="neither_continuous_nor_discrete",
        )
