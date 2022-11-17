"""Array manipulation functions."""

from typing import List, Optional, Sequence, Tuple, Union

from .. import BACKEND, Array, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

from .. import asshape
from ..typing import ShapeLike, ShapeType

__all__ = [
    "atleast_1d",
    "atleast_2d",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "concat",
    "expand_axes",
    "flip",
    "hstack",
    "move_axes",
    "permute_axes",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "swap_axes",
    "tile",
    "vstack",
]
__all__.sort()


def atleast_1d(*arrays: Array):
    """Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Parameters
    ----------
    arrays
        One or more input arrays.

    Returns
    -------
    out
        An array, or list of arrays, each with ``a.ndim >= 1``.

    See Also
    --------
    atleast_2d : Convert inputs to arrays with at least two dimensions.
    """
    return _impl.atleast_1d(*arrays)


def atleast_2d(*arrays: Array):
    """Convert inputs to arrays with at least two dimensions.

    Parameters
    ----------
    arrays
        One or more input arrays.

    Returns
    -------
    out
        An array, or list of arrays, each with ``a.ndim >= 2``.

    See Also
    --------
    atleast_1d : Convert inputs to arrays with at least one dimension.
    """
    return _impl.atleast_2d(*arrays)


def broadcast_arrays(*arrays: Array) -> List[Array]:
    """Broadcasts one or more arrays against one another.

    Parameters
    ----------
    arrays
        An arbitrary number of to-be broadcasted arrays.

    Returns
    -------
    out
        A list of broadcasted arrays.
    """
    return _impl.broadcast_arrays(*arrays)


def broadcast_shapes(*shapes: ShapeType) -> ShapeType:
    """Broadcast the input shapes into a single shape.

    Returns the resulting shape of `broadcasting
    <https://data-apis.org/array-api/latest/API_specification/broadcasting.html>`_
    arrays of the given ``shapes``.

    Parameters
    ----------
    shapes
        The shapes to be broadcast against each other.

    Returns
    -------
    outshape
        Broadcasted shape.
    """
    return _impl.broadcast_shapes(*shapes)


def broadcast_to(x: Array, /, shape: ShapeLike) -> Array:
    """Broadcasts an array to a specified shape.

    Parameters
    ----------
    x
        Array to broadcast.
    shape
        Array shape. Must be compatible with ``x``.

    Returns
    -------
    out
        An array having a specified shape.
    """
    return _impl.broadcast_to(x, shape=asshape(shape))


def concat(
    arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0
) -> Array:
    """Joins a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays
        Input arrays to join. The arrays must have the same shape, except in the
        dimension specified by ``axis``.
    axis
        Axis along which the arrays will be joined. If ``axis`` is ``None``, arrays are
        flattened before concatenation.

    Returns
    -------
    out
        An output array containing the concatenated values.
    """
    return _impl.concat(arrays, axis=axis)


def expand_axes(x: Array, /, *, axis: int = 0) -> Array:
    """Expands the shape of an array by inserting a new axis of size one at the position
    specified by ``axis``.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis position.

    Returns
    -------
    out
        An expanded output array having the same data type as ``x``.
    """
    return _impl.expand_axes(x, axis=axis)


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """Reverses the order of elements in an array along the given axis.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis (or axes) along which to flip. If ``axis`` is ``None``, the function will
        flip all input array axes.

    Returns
    -------
    out
        An output array having the same data type and shape as ``x`` and whose elements,
        relative to ``x``, are reordered.
    """
    return _impl.flip(x, axis=axis)


def permute_axes(x: Array, /, axes: Tuple[int, ...]) -> Array:
    """Permutes the axes of an array ``x``.

    Parameters
    ----------
    x
        input array.
    axes
        Tuple containing a permutation of ``(0, 1, ..., N-1)`` where ``N`` is the number
        of axes of ``x``.

    Returns
    -------
    out
        An array containing the axes permutation.

    See Also
    --------
    swap_axes : Permute the axes of an array.
    """
    return _impl.permute_axes(x, axes=axes)


def move_axes(
    x: Array,
    /,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> Array:
    """Move axes of an array to new positions.

    Other axes remain in the original order

    Parameters
    ----------
    x
        Array whose axes should be reordered.
    source
        Original positions of the axes to move. These must be unique.
    destination
        Destination positions for each of the original axes. These must also be unique.

    Returns
    -------
    out
        Array with moved axes.
    """
    return _impl.move_axes(x, source=source, destination=destination)


def swap_axes(x: Array, /, axis1: int, axis2: int) -> Array:
    """Swaps the axes of an array ``x``.

    Parameters
    ----------
    x
        Input array.
    axis1
        First axis to be swapped.
    axis2
        Second axis to be swapped.

    Returns
    -------
    out
        An array containing the swapped axes.

    See Also
    --------
    permute_axes : Permute the axes of an array.
    """
    return _impl.swap_axes(x, axis1=axis1, axis2=axis2)


def reshape(x: Array, /, shape: ShapeLike, *, copy: Optional[bool] = None) -> Array:
    """Reshapes an array without changing its data.

    Parameters
    ----------
    x
        Input array to reshape.
    shape
        A new shape compatible with the original shape. One shape dimension is allowed
        to be ``-1``. When a shape dimension is ``-1``, the corresponding output array
        shape dimension will be inferred from the length of the array and the remaining
        dimensions.
    copy
        Boolean indicating whether or not to copy the input array. If ``None``, reuses
        existing memory buffer if possible and copy otherwise.

    Returns
    -------
    out
        An output array having the same data type and elements as ``x``.
    """
    return _impl.reshape(x, shape=asshape(shape), copy=copy)


def roll(
    x: Array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Array:
    """Rolls array elements along a specified axis.

    Array elements that roll beyond the last position are re-introduced at the first
    position. Array elements that roll beyond the first position are re-introduced at
    the last position.

    Parameters
    ----------
    x
        Input array.
    shift
        Number of places by which the elements are shifted. If ``shift`` is a tuple,
        then ``axis`` must be a tuple of the same size, and each of the given axes will
        be shifted by the corresponding element in ``shift``. If ``shift`` is an ``int``
        and ``axis`` a tuple, then the same ``shift`` will be used for all specified
        axes.
    axis
        Axis (or axes) along which elements to shift. If ``axis`` is ``None``, the array
        will be flattened, shifted, and then restored to its original shape.

    Returns
    -------
    out
        An output array having the same data type as ``x`` and whose elements, relative
        to ``x``, are shifted.
    """
    return _impl.roll(x, shift=shift, axis=axis)


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    """Removes singleton axes from ``x``.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis (or axes) to squeeze.

    Returns
    -------
    out
        An output array having the same data type and elements as ``x``.
    """
    return _impl.squeeze(x, axis=axis)


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """Joins a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays
        Input arrays to join. Each array must have the same shape.
    axis
        Axis along which the arrays will be joined. Providing an ``axis`` specifies the
        index of the new axis in the dimensions of the result. For example, if ``axis``
        is ``0``, the new axis will be the first dimension and the output array will
        have shape ``(N, A, B, C)``; if ``axis`` is ``1``, the new axis will be the
        second dimension and the output array will have shape ``(A, N, B, C)``.

    Returns
    --------
    out
        An output array having rank ``N+1``, where ``N`` is the rank (number of
        dimensions) of ``x``.
    """
    return _impl.stack(arrays, axis=axis)


def hstack(arrays: Union[Tuple[Array, ...], List[Array]], /) -> Array:
    """Joins a sequence of arrays horizontally (column-wise).

    Parameters
    ----------
    arrays
        Input arrays to join. Each array must have the same shape along all but the
        second axis.

    Returns
    --------
    out
        An output array formed by stacking the given arrays.
    """
    return _impl.hstack(arrays)


def vstack(arrays: Union[Tuple[Array, ...], List[Array]], /) -> Array:
    """Joins a sequence of arrays vertically (column-wise).

    Parameters
    ----------
    arrays
        Input arrays to join. Each array must have the same shape along all but the
        first axis.

    Returns
    --------
    out
        An output array formed by stacking the given arrays.
    """
    return _impl.vstack(arrays)


def tile(A: Array, /, reps: ShapeLike) -> Array:
    """Construct an array by repeating ``A`` the number of times given by ``reps``.

    If ``reps`` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, ``A`` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote ``A`` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, ``reps`` is promoted to ``A``.ndim by pre-pending 1's to it.
    Thus for an ``A`` of shape (2, 3, 4, 5), a ``reps`` of (2, 2) is treated as
    (1, 1, 2, 2).

    .. note::

        Although tile may be used for broadcasting, it is strongly recommended to use
        broadcasting operations and functionality instead.

    Parameters
    ----------
    A
        The input array.
    reps
        The number of repetitions of ``A`` along each axis.

    Returns
    -------
    out
        The tiled output array.
    """
    return _impl.tile(A, asshape(reps))
