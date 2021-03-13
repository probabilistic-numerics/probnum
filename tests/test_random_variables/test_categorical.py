"""Tests for the categorical random variable."""


import string

import numpy as np
import pytest

from probnum import random_variables, utils

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
def probabilities():
    probabilities = np.random.rand(NDIM)
    return probabilities / np.sum(probabilities)


@pytest.fixture
def categ(probabilities, support, random_state):
    return random_variables.Categorical(
        probabilities=probabilities, support=support, random_state=random_state
    )


@all_random_states
@all_supports
def test_probabilities(categ, probabilities):
    assert categ.probabilities.shape == (NDIM,)
    np.testing.assert_allclose(categ.probabilities, probabilities)


@all_supports
@all_random_states
def test_support(categ):
    assert len(categ.support) == NDIM
    assert isinstance(categ.support, np.ndarray)


@all_supports
@all_random_states
@pytest.mark.parametrize("size", [(), 1, (1,), (1, 1)])
def test_sample(categ, size):
    samples = categ.sample(size=size)
    expected_shape = utils.as_shape(size) + categ.shape
    assert samples.shape == expected_shape


@all_supports
@all_random_states
@pytest.mark.parametrize("index", [0, -1])
def test_pmf(categ, index):
    pmf_value = categ.pmf(x=categ.support[index])
    np.testing.assert_almost_equal(pmf_value, categ.probabilities[index])

    zero_pmf_value = categ.pmf(x=np.inf * np.ones(categ.shape))
    np.testing.assert_almost_equal(zero_pmf_value, 0.0)


#
# @all_supports
# @all_random_states
# @pytest.mark.parametrize("index", [0, -1])
# def test_pmf_zero(categ, index):
#     zero_pmf_value = categ.pmf(x=np.inf * np.ones(categ.shape))
#     np.testing.assert_almost_equal(zero_pmf_value, 0.)
#
