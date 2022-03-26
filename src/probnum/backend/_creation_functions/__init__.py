"""Array creation functions."""


from .. import BACKEND, Array, Backend

if BACKEND is Backend.NUMPY:
    from . import _numpy as _core
elif BACKEND is Backend.JAX:
    from . import _jax as _core
elif BACKEND is Backend.TORCH:
    from . import _torch as _core

__all__ = ["tril", "triu"]


def tril(x: Array, /, *, k: int = 0) -> Array:
    """Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::

       The lower triangular part of the matrix is defined as the elements on and below
       the specified diagonal ``k``.

    Parameters
    ----------
    x
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices.
    k
        diagonal above which to zero elements. If ``k = 0``, the diagonal is the main
        diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``,
        the diagonal is above the main diagonal. Default: ``0``.

        .. note::

           The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on
           the interval ``[0, min(M, N) - 1]``.

    Returns
    -------
    out :
        an array containing the lower triangular part(s). The returned array must have
        the same shape and data type as ``x``. All elements above the specified diagonal
        ``k`` must be zeroed. The returned array should be allocated on the same device
        as ``x``.
    """
    return _core.tril(x, k=k)


def triu(x: Array, /, *, k: int = 0) -> Array:
    """Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::

       The upper triangular part of the matrix is defined as the elements on and above
       the specified diagonal ``k``.

    Parameters
    ----------
    x
        Input array having shape ``(..., M, N)`` and whose innermost two dimensions form
        ``MxN`` matrices.
    k
        Diagonal below which to zero elements. If ``k = 0``, the diagonal is the main
        diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``,
        the diagonal is above the main diagonal. Default: ``0``.

        .. note::

           The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on
           the interval ``[0, min(M, N) - 1]``.

    Returns
    -------
    out:
        An array containing the upper triangular part(s). The returned array must have
        the same shape and data type as ``x``. All elements below the specified diagonal
        ``k`` must be zeroed. The returned array should be allocated on the same device
        as ``x``.
    """
    return _core.triu(x, k=k)
