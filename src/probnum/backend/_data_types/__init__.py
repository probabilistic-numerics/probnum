"""Data types."""

from __future__ import annotations

from .. import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = [
    "Dtype",
    "bool",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

Dtype = _impl.Dtype
bool = _impl.bool
int32 = _impl.int32
int64 = _impl.int64
float16 = _impl.float16
float32 = _impl.float32
float64 = _impl.float64
complex64 = _impl.complex64
complex128 = _impl.complex128
