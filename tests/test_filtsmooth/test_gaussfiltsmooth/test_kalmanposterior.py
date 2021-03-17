import numpy as np
import pytest

import probnum.filtsmooth as pnfs
import probnum.statespace as pnss
from probnum import randvars
from probnum._randomvariablelist import _RandomVariableList

from .filtsmooth_testcases import car_tracking


@pytest.fixture
def problem():
    """Car-tracking problem."""
    problem = car_tracking()
    dynmod, measmod, initrv, info = problem

    times = np.arange(0, info["tmax"], info["dt"])
    states, obs = pnss.generate_samples(
        dynmod=dynmod, measmod=measmod, initrv=initrv, times=times
    )
    return dynmod, measmod, initrv, info, obs, times, states


@pytest.fixture
def kalman(problem):
    """Standard Kalman filter."""
    dynmod, measmod, initrv, *_ = problem
    return pnfs.Kalman(dynmod, measmod, initrv)


@pytest.fixture
def posterior(kalman, problem):
    """Kalman smoothing posterior."""
    *_, obs, times, states = problem
    return kalman.filtsmooth(obs, times)


def test_len(posterior):
    """__len__ performs as expected."""
    assert len(posterior) > 0
    assert len(posterior.locations) == len(posterior)
    assert len(posterior.state_rvs) == len(posterior)


def test_locations(posterior, problem):
    """Locations are stored correctly."""
    *_, obs, times, states = problem
    np.testing.assert_allclose(posterior.locations, np.sort(posterior.locations))
    np.testing.assert_allclose(posterior.locations, times)


def test_getitem(posterior):
    """Getitem performs as expected."""

    np.testing.assert_allclose(posterior[0].mean, posterior.state_rvs[0].mean)
    np.testing.assert_allclose(posterior[0].cov, posterior.state_rvs[0].cov)

    np.testing.assert_allclose(posterior[-1].mean, posterior.state_rvs[-1].mean)
    np.testing.assert_allclose(posterior[-1].cov, posterior.state_rvs[-1].cov)

    np.testing.assert_allclose(posterior[:].mean, posterior.state_rvs[:].mean)
    np.testing.assert_allclose(posterior[:].cov, posterior.state_rvs[:].cov)


def test_state_rvs(posterior):
    """RVs are stored correctly."""

    assert isinstance(posterior.state_rvs, _RandomVariableList)
    assert len(posterior.state_rvs[0].shape) == 1


def test_call_error_if_small(posterior):
    """Evaluating in the past of the data raises an error."""
    assert -0.5 < posterior.locations[0]
    with pytest.raises(ValueError):
        posterior(-0.5)


def test_call_vectorisation(posterior):
    """Evaluation allows vector inputs."""
    locs = np.arange(0, 1, 20)
    evals = posterior(locs)
    assert len(evals) == len(locs)


def test_call_interpolation(posterior):
    """Interpolation is possible and returns a Normal RV."""
    assert posterior.locations[0] < 9.88 < posterior.locations[-1]
    assert 9.88 not in posterior.locations
    out_rv = posterior(9.88)
    assert isinstance(out_rv, randvars.Normal)


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
    """Extrapolation is possible and returns a Normal RV."""
    assert posterior.locations[-1] < 30.0
    out_rv = posterior(30.0)
    assert isinstance(out_rv, randvars.Normal)


@pytest.fixture
def size():
    return 10


@pytest.fixture
def size():
    return (12,)


@pytest.fixture
def locs(posterior):
    return None


@pytest.fixture
def locs(posterior):
    return posterior.locations[[2, 3]]


@pytest.fixture
def locs(posterior):
    return np.arange(0, 0.5, 0.025)


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def samples(posterior, locs, size, seed):
    return posterior.sample(locations=locs, size=size)


def test_sampling_shapes(samples, posterior, locs, size):
    """Shape of the returned samples matches expectation."""
    if isinstance(size, int):
        size = (size,)
    expected_size = (*size, *posterior(locs).mean.shape)
    assert samples.shape == expected_size
