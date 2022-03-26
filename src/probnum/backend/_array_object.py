"""Basic class representing an array."""

import probnum.backend as _backend

if _backend.BACKEND is _backend.Backend.NUMPY:
    from numpy import ndarray as Array
elif _backend.BACKEND is _backend.Backend.JAX:
    from jax.numpy import ndarray as Array
elif _backend.BACKEND is _backend.Backend.TORCH:
    from torch import Tensor as Array

__all__ = ["Array"]
