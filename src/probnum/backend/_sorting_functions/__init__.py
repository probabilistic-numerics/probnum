"""Sorting functions."""

from .. import BACKEND, Array, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _core
elif BACKEND is Backend.JAX:
    from . import _jax as _core
elif BACKEND is Backend.TORCH:
    from . import _torch as _core

__all__ = ["argsort", "sort"]


def argsort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    """Returns the indices that sort an array ``x`` along a specified axis.

    Parameters
    ----------
    x
        input array.
    axis
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending
        sort order. If ``True``, the returned indices sort ``x`` in descending order
        (by value). If ``False``, the returned indices sort ``x`` in ascending order
        (by value). Default: ``False``.
    stable
        sort stability. If ``True``, the returned indices must maintain the relative
        order of ``x`` values which compare as equal. If ``False``, the returned indices
        may or may not maintain the relative order of ``x`` values which compare as
        equal (i.e., the relative order of ``x`` values which compare as equal is
        implementation-dependent). Default: ``True``.

    Returns
    -------
    out :
        an array of indices. The returned array must have the same shape as ``x``. The
        returned array must have the default array index data type.
    """
    return _core.argsort(x, axis=axis, descending=descending, stable=stable)


def sort(
    x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> Array:
    """Returns a sorted copy of an input array ``x``.

    Parameters
    ----------
    x
        input array.
    axis
        axis along which to sort. If set to ``-1``, the function must sort along the
        last axis. Default: ``-1``.
    descending
        sort order. If ``True``, the array must be sorted in descending order (by
        value). If ``False``, the array must be sorted in ascending order (by value).
        Default: ``False``.
    stable
        sort stability. If ``True``, the returned array must maintain the relative order
        of ``x`` values which compare as equal. If ``False``, the returned array may or
        may not maintain the relative order of ``x`` values which compare as equal
        (i.e., the relative order of ``x`` values which compare as equal is
        implementation-dependent). Default: ``True``.
    Returns
    -------
    out :
        a sorted array. The returned array must have the same data type and shape as
        ``x``.
    """
    return _core.sort(x, axis=axis, descending=descending, stable=stable)
