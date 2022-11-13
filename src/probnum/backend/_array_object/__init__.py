"""Array object."""

from __future__ import annotations

from typing import Any

from .. import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = ["Array", "Device", "Scalar", "isarray"]

Scalar = _impl.Scalar
Array = _impl.Array
Device = _impl.Device


def isarray(x: Any) -> bool:
    """Check whether an object is an :class:`~probnum.backend.Array`."""
    return isinstance(x, (Array, Scalar))
