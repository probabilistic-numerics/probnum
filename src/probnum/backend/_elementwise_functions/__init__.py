"""Elementwise functions."""

from .. import BACKEND, Array, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

__all__ = ["isnan"]


def isnan(x: Array, /) -> Array:
    """Tests each element ``x_i`` of the input array ``x`` to determine whether the
    element is ``NaN``.

    Parameters
    ----------
    x
        Input array. Should have a numeric data type.

    Returns
    -------
    out
        An array containing test results. An element ``out_i`` is ``True`` if ``x_i`` is
        ``NaN`` and ``False`` otherwise. The returned array should have a data type of
        ``bool``.
    """
    return _impl.isnan(x)
