"""Test fixtures for linear algebra."""

import numpy as np
import pytest_cases


@pytest_cases.fixture()
def rng() -> np.random.Generator:
    """Random number generator."""
    return np.random.default_rng(42)
