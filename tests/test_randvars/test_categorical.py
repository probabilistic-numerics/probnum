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


@pytest.fixture
def rng():
    return np.random.default_rng(seed=123)


@pytest.fixture
def probabilities(rng):
    probabilities = rng.uniform(size=NDIM)
    return probabilities / np.sum(probabilities)


@pytest.fixture
def categ(probabilities, support):
    return randvars.Categorical(probabilities=probabilities, support=support)


@all_supports
def test_probabilities(categ, probabilities):
    assert categ.probabilities.shape == (NDIM,)
    np.testing.assert_allclose(categ.probabilities, probabilities)


@all_supports
def test_support(categ):
    assert len(categ.support) == NDIM
    assert isinstance(categ.support, np.ndarray)


@all_supports
@pytest.mark.parametrize("size", [(), 1, (1,), (1, 1)])
def test_sample(categ, size, rng):
    samples = categ.sample(rng=rng, size=size)
    expected_shape = utils.as_shape(size) + categ.shape
    assert samples.shape == expected_shape


@all_supports
@pytest.mark.parametrize("index", [0, -1])
def test_pmf(categ, index):
    pmf_value = categ.pmf(x=categ.support[index])
    np.testing.assert_almost_equal(pmf_value, categ.probabilities[index])


@all_supports
def test_pmf_zero(categ):
    """Make a new Categorical RV that excludes the final point and check that the pmf
    rightfully evaluates to zero."""

    new_categ = randvars.Categorical(
        support=categ.support[:-1],
        probabilities=categ.probabilities[:-1],
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
def test_mode(categ):
    mode = categ.mode
    assert mode.shape == categ.shape


@all_supports
def test_resample(categ, rng):
    new_categ = categ.resample(rng=rng)

    assert isinstance(new_categ, randvars.Categorical)
    assert new_categ.shape == categ.shape

    # Assert all probabilities are equal
    np.testing.assert_allclose(np.diff(new_categ.probabilities), 0.0)

    # Assert support is chosen from old support
    assert np.all(np.isin(new_categ.support, categ.support))
