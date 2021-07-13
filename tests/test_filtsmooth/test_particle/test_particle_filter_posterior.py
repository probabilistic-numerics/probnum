import numpy as np
import pytest

from probnum import filtsmooth, randvars


@pytest.fixture
def state_list():
    return [
        randvars.Categorical(
            probabilities=np.ones(10) / 10, support=np.random.rand(10, 2)
        )
        for _ in range(20)
    ]


@pytest.fixture
def posterior(state_list):
    return filtsmooth.particle.ParticleFilterPosterior(
        states=state_list, locations=np.arange(20)
    )


def test_len(posterior):
    assert len(posterior) == 20


def test_getitem(posterior):
    assert len(posterior) == 20


def test_call(posterior):
    with pytest.raises(NotImplementedError):
        posterior(0.0)


# The tests below -- as a side results -- also cover the specific properties in _RandomVariableList.


def test_mode(posterior):
    mode = posterior.states.mode
    assert mode.shape == (len(posterior),) + posterior.states[0].shape


def test_support(posterior):
    support = posterior.states.support
    assert (
        support.shape
        == (len(posterior),)
        + (len(posterior.states[0].probabilities),)
        + posterior.states[0].shape
    )


def test_probabilities(posterior):
    probabilities = posterior.states.probabilities
    assert probabilities.shape == (len(posterior),) + (
        len(posterior.states[0].probabilities),
    )


def test_resample(posterior, rng):
    resampled_posterior = posterior.states.resample(rng=rng)
    assert (
        resampled_posterior.support.shape
        == (len(posterior),)
        + (len(posterior.states[0].probabilities),)
        + posterior.states[0].shape
    )
