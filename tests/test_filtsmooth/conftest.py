"""Test fixtures for filtering and smoothing."""


import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)
