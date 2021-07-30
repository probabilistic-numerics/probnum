import numpy as np
import pytest

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq, randvars
from probnum._randomvariablelist import _RandomVariableList


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def stepsize():
    return 0.1


@pytest.fixture
def timespan():
    return (0.0, 0.5)


@pytest.fixture
def posterior(stepsize, timespan):
    """Kalman smoothing posterior."""
    t0, tmax = timespan
    y0 = 20 * np.ones(2)
    ivp = diffeq_zoo.lotkavolterra(t0=t0, tmax=tmax, y0=y0)
    f = ivp.f
    t0, tmax = ivp.t0, ivp.tmax
    return diffeq.probsolve_ivp(f, t0, tmax, y0, step=stepsize, adaptive=False)


def test_len(posterior):
    """__len__ performs as expected."""
    assert len(posterior) > 0
    assert len(posterior.locations) == len(posterior)
    assert len(posterior.states) == len(posterior)


def test_locations(posterior, stepsize, timespan):
    """Locations are stored correctly."""
    np.testing.assert_allclose(posterior.locations, np.sort(posterior.locations))

    t0, tmax = timespan
    expected = np.arange(t0, tmax + stepsize, step=stepsize)
    np.testing.assert_allclose(posterior.locations, expected)


def test_getitem(posterior):
    """Getitem performs as expected."""

    np.testing.assert_allclose(posterior[0].mean, posterior.states[0].mean)
    np.testing.assert_allclose(posterior[0].cov, posterior.states[0].cov)

    np.testing.assert_allclose(posterior[-1].mean, posterior.states[-1].mean)
    np.testing.assert_allclose(posterior[-1].cov, posterior.states[-1].cov)

    np.testing.assert_allclose(posterior[:].mean, posterior.states[:].mean)
    np.testing.assert_allclose(posterior[:].cov, posterior.states[:].cov)


def test_states(posterior):
    """RVs are stored correctly."""

    assert isinstance(posterior.states, _RandomVariableList)
    assert len(posterior.states[0].shape) == 1


def test_call_error_if_small(posterior):
    """Evaluating in the past of the data raises an error."""
    assert -0.5 < posterior.locations[0]
    with pytest.raises(NotImplementedError):
        posterior(-0.5)


def test_call_vectorisation(posterior):
    """Evaluation allows vector inputs."""
    locs = np.arange(0, 1, 20)
    evals = posterior(locs)
    assert len(evals) == len(locs)


def test_call_interpolation(posterior):
    """Interpolation is possible and returns a Normal RV."""

    a = 0.4 + 0.1 * np.random.rand()
    t0, t1 = posterior.locations[:2]
    random_location_between_t0_and_t1 = t0 + a * (t1 - t0)
    assert (
        posterior.locations[0]
        < random_location_between_t0_and_t1
        < posterior.locations[-1]
    )
    assert random_location_between_t0_and_t1 not in posterior.locations
    out_rv = posterior(random_location_between_t0_and_t1)
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
def seed():
    return 42


# Sampling shape checks include extrapolation phases
IN_DOMAIN_DENSE_LOCS = np.arange(0.0, 0.5, 0.025)
OUT_OF_DOMAIN_DENSE_LOCS = np.arange(0.0, 500.0, 25.0)


@pytest.mark.parametrize("locs", [None, IN_DOMAIN_DENSE_LOCS, OUT_OF_DOMAIN_DENSE_LOCS])
@pytest.mark.parametrize("size", [(), 2, (2,), (2, 2)])
def test_sampling_shapes(posterior, locs, size, rng):
    """Shape of the returned samples matches expectation."""
    samples = posterior.sample(rng=rng, t=locs, size=size)

    if isinstance(size, int):
        size = (size,)
    if locs is None:
        expected_size = (
            size + posterior.states.shape
        )  # (*size, *posterior.states.shape)
    else:
        expected_size = (
            size + locs.shape + posterior.states[0].shape
        )  # (*size, *posterior(locs).mean.shape)

    assert samples.shape == expected_size


def test_transform_base_measure_realizations_raises_error(posterior):
    """The KalmanODESolution does not implement transformation of base measure
    realizations, but refers to KalmanPosterior instead."""
    with pytest.raises(NotImplementedError):
        posterior.transform_base_measure_realizations(None)
