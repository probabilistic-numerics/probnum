import pytest

from probnum import backend, compat


@pytest.mark.parametrize("scalar", [1, 1.0, 1.0 + 2.0j, backend.array(1.0)])
def test_as_scalar_returns_scalar_array(scalar):
    """All sorts of scalars are transformed into a np.generic."""
    as_scalar = backend.as_scalar(scalar)
    assert isinstance(as_scalar, backend.ndarray) and as_scalar.shape == ()
    compat.testing.assert_allclose(as_scalar, scalar, atol=0.0, rtol=1e-12)


@pytest.mark.parametrize("sequence", [[1.0], (1,), backend.array([1.0])])
def test_as_scalar_sequence_error(sequence):
    """Sequence types give rise to ValueErrors in `as_scalar`."""
    with pytest.raises(ValueError):
        backend.as_scalar(sequence)
