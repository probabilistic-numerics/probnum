"""Test for the convenience functions."""

import numpy as np
import pytest

from probnum import filtsmooth


@pytest.fixture
def prior_dimension():
    return 3


@pytest.fixture
def measurement_dimension():
    return 2


@pytest.fixture
def number_of_measurements():
    return 10


@pytest.fixture
def observations(measurement_dimension, number_of_measurements):
    obs = np.arange(measurement_dimension * number_of_measurements)
    return obs.reshape((number_of_measurements, measurement_dimension))


@pytest.fixture
def locations(number_of_measurements):
    return np.arange(number_of_measurements)


@pytest.fixture
def F(prior_dimension):
    return np.eye(prior_dimension)


@pytest.fixture
def L(prior_dimension):
    return np.eye(prior_dimension)


@pytest.fixture
def H(measurement_dimension, prior_dimension):
    return np.eye(measurement_dimension, prior_dimension)


@pytest.fixture
def R(measurement_dimension):
    return np.eye(measurement_dimension)


@pytest.fixture
def m0(prior_dimension):
    return np.arange(prior_dimension)


@pytest.fixture
def C0(prior_dimension):
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
