from probnum import backend

import pytest


@pytest.mark.parametrize(
    "x", [backend.random.uniform(backend.random.rng_state(42), shape=(5, 2, 6))]
)
def test_diagonal_acts_on_last_axes(x: backend.Array):
    assert x.shape[:-2] == backend.linalg.diagonal(x).shape[:-1]
