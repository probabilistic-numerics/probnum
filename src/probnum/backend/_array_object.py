"""Basic class representing an array."""

from typing import Any

import probnum.backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from numpy import generic as Scalar, ndarray as Array
elif _backend.BACKEND is _backend.Backend.JAX:
    from jax.numpy import ndarray as Array, ndarray as Scalar
elif _backend.BACKEND is _backend.Backend.TORCH:
    from torch import Tensor as Array, Tensor as Scalar


__all__ = ["Scalar", "Array", "isarray"]


def isarray(x: Any) -> bool:
    return isinstance(x, (Array, Scalar))
