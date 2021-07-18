"""Configuration for tests."""

import pytest


@pytest.fixture
def some_ordint(test_ndim):
    return test_ndim - 1
