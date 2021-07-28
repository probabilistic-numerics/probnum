"""Tests for the coordinate conversion functions."""

import numpy as np
import pytest

from probnum import randprocs


@pytest.fixture
def some_order():
    return 2


@pytest.fixture
def some_dim():
    return 3


@pytest.fixture
def fake_state(some_order, some_dim):
    return np.arange(some_dim * (some_order + 1))


@pytest.fixture
def in_out_pair(fake_state, some_order):
    """Initial states for the three-body initial values.

    Returns (derivwise, coordwise)
    """
    return fake_state, fake_state.reshape((-1, some_order + 1)).T.flatten()


def test_in_out_pair_is_not_identical(in_out_pair):
    """A little sanity check to assert that these are actually different, so the
    conversion is non-trivial."""
    derivwise, coordwise = in_out_pair
    assert np.linalg.norm(derivwise - coordwise) > 5


def test_convert_coordwise_to_derivwise(in_out_pair, some_order, some_dim):
    derivwise, coordwise = in_out_pair
    coordwise_as_derivwise = (
        randprocs.markov.integrator.convert.convert_coordwise_to_derivwise(
            coordwise, some_order, some_dim
        )
    )
    np.testing.assert_allclose(coordwise_as_derivwise, derivwise)


def test_convert_derivwise_to_coordwise(in_out_pair, some_order, some_dim):
    derivwise, coordwise = in_out_pair
    derivwise_as_coordwise = (
        randprocs.markov.integrator.convert.convert_derivwise_to_coordwise(
            derivwise, some_order, some_dim
        )
    )
    np.testing.assert_allclose(derivwise_as_coordwise, coordwise)


def test_conversion_pairwise_inverse(in_out_pair, some_order, some_dim):
    derivwise, coordwise = in_out_pair
    as_coord = randprocs.markov.integrator.convert.convert_derivwise_to_coordwise(
        derivwise, some_order, some_dim
    )
    as_deriv_again = randprocs.markov.integrator.convert.convert_coordwise_to_derivwise(
        as_coord, some_order, some_dim
    )
    np.testing.assert_allclose(as_deriv_again, derivwise)
