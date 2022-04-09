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
    "float32",
    "float64",
    "complex64",
    "complex128",
]

Dtype = _impl.Dtype
"""Data type of an array."""

bool = _impl.bool
"""Boolean (``True`` or ``False``)."""

int32 = _impl.int32
"""A 32-bit signed integer whose values exist on the interval
``[-2,147,483,647, +2,147,483,647]``."""

int64 = _impl.int64
"""A 64-bit signed integer whose values exist on the interval
``[-9,223,372,036,854,775,807, +9,223,372,036,854,775,807]``."""

float16 = _impl.float16
"""IEEE 754 half-precision (16-bit) binary floating-point number (see IEEE 754-2019).
"""

float32 = _impl.float32
"""IEEE 754 single-precision (32-bit) binary floating-point number (see IEEE 754-2019).
"""

float64 = _impl.float64
"""IEEE 754 double-precision (64-bit) binary floating-point number (see IEEE 754-2019).
"""

complex64 = _impl.complex64
"""Single-precision complex number represented by two single-precision floats (real and
imaginary components."""

complex128 = _impl.complex128
"""Double-precision complex number represented by two double-precision floats (real and
imaginary components."""
