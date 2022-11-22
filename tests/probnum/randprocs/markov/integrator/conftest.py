"""Configuration for tests."""

import pytest


@pytest.fixture
def some_num_derivatives(state_dim):
    return state_dim - 1
