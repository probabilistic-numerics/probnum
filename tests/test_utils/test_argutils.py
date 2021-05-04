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


@pytest.mark.parametrize("shape_arg", list(range(5)) + [np.int32(8)])
@pytest.mark.parametrize("ndim", [False, True])
def test_as_shape_int(shape_arg, ndim):
    if ndim:
        shape = pnut.as_shape(shape_arg, ndim=1)
    else:
        shape = pnut.as_shape(shape_arg)

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
        shape = pnut.as_shape(shape_arg, ndim=len(shape_arg))
    else:
        shape = pnut.as_shape(shape_arg)

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
        pnut.as_shape(shape_arg)


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
        pnut.as_shape(shape_arg, ndim=ndim)
