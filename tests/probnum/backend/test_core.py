import numpy as np
import pytest

from probnum import backend, compat


@pytest.mark.parametrize("shape_arg", list(range(5)) + [np.int32(8)])
@pytest.mark.parametrize("ndim", [False, True])
def test_as_shape_int(shape_arg, ndim):
    if ndim:
        shape = backend.as_shape(shape_arg, ndim=1)
    else:
        shape = backend.as_shape(shape_arg)

    assert isinstance(shape, tuple)
    assert len(shape) == 1
    assert all(isinstance(entry, int) for entry in shape)
    assert shape[0] == shape_arg


@pytest.mark.parametrize(
    "shape_arg",
    [
        (),
        [],
        (2,),
        [3],
        [3, 6, 5],
        (1, 1, 1),
        (np.int32(7), 2, 4, 8),
    ],
)
@pytest.mark.parametrize("ndim", [False, True])
def test_as_shape_iterable(shape_arg, ndim):
    if ndim:
        shape = backend.as_shape(shape_arg, ndim=len(shape_arg))
    else:
        shape = backend.as_shape(shape_arg)

    assert isinstance(shape, tuple)
    assert len(shape) == len(shape_arg)
    assert all(isinstance(entry, int) for entry in shape)
    assert all(
        entry_shape == entry_shape_arg
        for entry_shape, entry_shape_arg in zip(shape_arg, shape)
    )


@pytest.mark.parametrize(
    "shape_arg",
    [
        None,
        "(1, 2, 3)",
        tuple,
    ],
)
def test_as_shape_wrong_type(shape_arg):
    with pytest.raises(TypeError):
        backend.as_shape(shape_arg)


@pytest.mark.parametrize(
    "shape_arg, ndim",
    [
        ((), 1),
        ([], 4),
        (3, 3),
        ((2,), 8),
        ([3], 5),
        ([3, 6, 5], 2),
        ((1, 1, 1), 5),
        ((np.int32(7), 2, 4, 8), 2),
    ],
)
def test_as_shape_wrong_ndim(shape_arg, ndim):
    with pytest.raises(TypeError):
        backend.as_shape(shape_arg, ndim=ndim)


@pytest.mark.parametrize("scalar", [1, 1.0, 1.0 + 2.0j, backend.array(1.0)])
def test_asscalar_returns_scalar_array(scalar):
    """All sorts of scalars are transformed into a np.generic."""
    asscalar = backend.asscalar(scalar)
    assert backend.isarray(asscalar) and asscalar.shape == ()
    compat.testing.assert_allclose(asscalar, scalar, atol=0.0, rtol=1e-12)


@pytest.mark.parametrize("sequence", [[1.0], (1,), backend.array([1.0])])
def test_asscalar_sequence_error(sequence):
    """Sequence types give rise to ValueErrors in `asscalar`."""
    with pytest.raises(ValueError):
        backend.asscalar(sequence)
