"""Array object."""

from __future__ import annotations

from typing import Any

from .. import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _core
elif BACKEND is Backend.JAX:
    from . import _jax as _core
elif BACKEND is Backend.TORCH:
    from . import _torch as _core

__all__ = ["Scalar", "Array", "dtype", "isarray"]

Scalar = _core.Scalar
Array = _core.Array
dtype = _core.dtype


def isarray(x: Any) -> bool:
    return isinstance(x, (Array, Scalar))
