"""Array object."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from ..._select_backend import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl


__all__ = ["asshape", "isarray", "ndim", "to_numpy", "Array", "Device", "Scalar"]

Scalar = _impl.Scalar
Array = _impl.Array
Device = _impl.Device


def asshape(
    x: "probnum.backend.typing.ShapeLike",
    ndim: Optional["probnum.backend.typing.IntLike"] = None,
) -> "probnum.backend.typing.ShapeType":
    """Convert a shape representation into a shape defined as a tuple of ints.

    Parameters
    ----------
    x
        Shape representation.
    ndim
        Number of axes / dimensions of the object with shape ``x``.

    Returns
    -------
    shape
        The input ``x`` converted to a :class:`~probnum.backend.typing.ShapeType`.

    Raises
    ------
    TypeError
        If the given ``x`` cannot be converted to a shape with ``ndim`` axes.
    """

    try:
        # x is an `IntLike`
        shape = (int(x),)
    except (TypeError, ValueError):
        # x is an iterable
        try:
            shape = tuple(int(item) for item in x)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"The given shape {x} must be an integer or an iterable of integers."
            ) from err

    if ndim is not None:
        ndim = int(ndim)

        if len(shape) != ndim:
            raise TypeError(f"The given shape {shape} must have {ndim} dimensions.")

    return shape


def isarray(x: Any) -> bool:
    """Check whether an object is an :class:`~probnum.backend.Array`.

    Parameters
    ----------
    x
       Object to check.
    """
    return isinstance(x, (Array, Scalar))


def ndim(x: Array) -> int:
    """Number of dimensions (axes) of an array.

    Parameters
    ----------
    x
        Array to get dimensions of.
    """
    return _impl.ndim(x)


def to_numpy(*arrays: Array) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Convert an :class:`~probnum.backend.Array` to a NumPy :class:`~numpy.ndarray`.

    Parameters
    ----------
    arrays
        Arrays to convert.
    """
    return _impl.to_numpy(*arrays)
