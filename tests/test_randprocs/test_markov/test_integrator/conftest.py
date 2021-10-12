"""Configuration for tests."""

import pytest


@pytest.fixture
def some_num_derivatives(test_ndim):
    return test_ndim - 1
