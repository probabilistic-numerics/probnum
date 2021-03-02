"""Basic tests for argutils."""

import numpy as np
import pytest

import probnum.utils as pnut


@pytest.mark.parametrize("scalar", [1, 1.0, 1.0 + 2.0j, np.array(1.0)])
def test_as_numpy_scalar_scalar_is_good(scalar):
    """All sorts of scalars are transformed into a np.generic."""
    as_scalar = pnut.as_numpy_scalar(scalar)
    assert isinstance(as_scalar, np.generic)
    np.testing.assert_allclose(as_scalar, scalar, atol=0.0, rtol=1e-12)


@pytest.mark.parametrize("sequence", [[1.0], (1,), np.array([1.0])])
def test_as_numpy_scalar_bad_sequence_is_bad(sequence):
    """Sequence types give rise to ValueErrors in `as_numpy_scalar`."""
    with pytest.raises(ValueError):
        pnut.as_numpy_scalar(sequence)
