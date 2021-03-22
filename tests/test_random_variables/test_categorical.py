"""Tests for the categorical random variable."""


import string

import numpy as np
import pytest

from probnum import randvars, utils

NDIM = 5

all_supports = pytest.mark.parametrize(
    "support",
    [
        None,
        np.arange(NDIM),
        np.array(list(string.ascii_lowercase)[:NDIM]),
        np.random.rand(NDIM, 3, 3),
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
    return randvars.Categorical(
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


@all_supports
@all_random_states
def test_pmf_zero(categ):
    """Make a new Categorical RV that excludes the final point and check that the pmf
    rightfully evaluates to zero."""

    new_categ = randvars.Categorical(
        support=categ.support[:-1],
        probabilities=categ.probabilities[:-1],
        random_state=categ.random_state,
    )
    zero_pmf_value = new_categ.pmf(x=categ.support[-1])
    np.testing.assert_almost_equal(zero_pmf_value, 0.0)


def test_pmf_valueerror():
    """If a PMF has string-valued support, its pmf cannot be evaluated at an integer.

    This value error is intended to guard against the issue presented in
    https://stackoverflow.com/questions/45020217/numpy-where-function-throws-a-futurewarning-returns-scalar-instead-of-list
    """
    categ = randvars.Categorical(probabilities=[0.5, 0.5], support=["a", "b"])
    with pytest.raises(ValueError):
        categ.pmf(2)


@all_supports
@all_random_states
def test_mode(categ):
    mode = categ.mode
    assert mode.shape == categ.shape


@all_supports
@all_random_states
def test_resample(categ):
    new_categ = categ.resample()

    assert isinstance(new_categ, randvars.Categorical)
    assert new_categ.shape == categ.shape

    # Assert all probabilities are equal
    np.testing.assert_allclose(np.diff(new_categ.probabilities), 0.0)

    # Assert support is chosen from old support
    assert np.all(np.isin(new_categ.support, categ.support))
