"""Configuration for tests."""

import pytest


@pytest.fixture
def some_nu(test_ndim):
    return test_ndim - 1
