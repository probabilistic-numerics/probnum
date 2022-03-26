"""Linear algebra."""

from .. import BACKEND, Array, Backend

__all__ = [
    "LinAlgError",
    "norm",
    "induced_norm",
    "inner_product",
    "gram_schmidt",
    "modified_gram_schmidt",
    "double_gram_schmidt",
    "cholesky",
    "solve",
    "solve_triangular",
    "solve_cholesky",
    "cholesky_update",
    "tril_to_positive_tril",
    "qr",
    "svd",
    "eigh",
]

if BACKEND is Backend.NUMPY:
    from . import _numpy as _core
elif BACKEND is Backend.JAX:
    from . import _jax as _core
elif BACKEND is Backend.TORCH:
    from . import _torch as _core

from numpy.linalg import LinAlgError

from ._cholesky_updates import cholesky_update, tril_to_positive_tril
from ._inner_product import induced_norm, inner_product
from ._orthogonalize import double_gram_schmidt, gram_schmidt, modified_gram_schmidt

norm = _core.norm
cholesky = _core.cholesky
solve_triangular = _core.solve_triangular
solve_cholesky = _core.solve_cholesky
qr = _core.qr
svd = _core.svd
eigh = _core.eigh


def solve(x1: Array, x2: Array, /) -> Array:
    """Returns the solution to the system of linear equations represented by the
    well-determined (i.e., full rank) linear matrix equation ``AX = B``.

    .. note::

       Whether an array library explicitly checks whether an input array is full rank is
       implementation-defined.

    Parameters
    ----------
    x1
        coefficient array ``A`` having shape ``(..., M, M)`` and whose innermost two
        dimensions form square matrices. Must be of full rank (i.e., all rows or,
        equivalently, columns must be linearly independent). Should have a
        floating-point data type.
    x2
        ordinate (or "dependent variable") array ``B``. If ``x2`` has shape ``(M,)``,
        ``x2`` is equivalent to an array having shape ``(..., M, 1)``. If ``x2`` has
        shape ``(..., M, K)``, each column ``k`` defines a set of ordinate values for
        which to compute a solution, and ``shape(x2)[:-1]`` must be compatible with
        ``shape(x1)[:-1]`` (see :ref:`broadcasting`). Should have a floating-point data
        type.

    Returns
    -------
    out:
        an array containing the solution to the system ``AX = B`` for each square
        matrix. The returned array must have the same shape as ``x2`` (i.e., the array
        corresponding to ``B``) and must have a floating-point data type determined by
        :ref:`type-promotion`.
    """
    return _core.solve(x1, x2)
