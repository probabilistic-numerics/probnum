"""Tests for the categorical random variable."""


import string

import numpy as np
import pytest

from probnum import random_variables

NDIM = 5

all_supports = pytest.mark.parametrize(
    "support",
    [
        None,
        np.arange(NDIM),
        np.array(list(string.ascii_lowercase)[:NDIM]),
        np.random.rand(NDIM, 3),
    ],
)

all_random_states = pytest.mark.parametrize(
    "random_state",
    [
        None,
        1,
        np.random.default_rng(),
    ],
)


@pytest.fixture
def event_probs():
    event_probs = np.random.rand(NDIM)
    return event_probs / np.sum(event_probs)


@pytest.fixture
def categ(event_probs, support, random_state):
    return random_variables.Categorical(
        event_probabilities=event_probs, support=support, random_state=random_state
    )


@all_random_states
@all_supports
def test_event_probabilities(categ, event_probs):
    assert categ.event_probabilities.shape == (NDIM,)
    np.testing.assert_allclose(categ.event_probabilities, event_probs)


@all_supports
@all_random_states
def test_support(categ):
    assert len(categ.support) == NDIM
    assert isinstance(categ.support, np.ndarray)


@all_supports
@all_random_states
def test_sample(categ):
    categ.sample()
