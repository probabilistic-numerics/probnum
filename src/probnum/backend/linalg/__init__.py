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
    from . import _numpy as _impl
elif BACKEND is Backend.JAX:
    from . import _jax as _impl
elif BACKEND is Backend.TORCH:
    from . import _torch as _impl

from numpy.linalg import LinAlgError

from ._cholesky_updates import cholesky_update, tril_to_positive_tril
from ._inner_product import induced_norm, inner_product
from ._orthogonalize import double_gram_schmidt, gram_schmidt, modified_gram_schmidt

norm = _impl.norm
cholesky = _impl.cholesky
solve_triangular = _impl.solve_triangular
solve_cholesky = _impl.solve_cholesky
qr = _impl.qr
svd = _impl.svd
eigh = _impl.eigh


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
        ``shape(x1)[:-1]`` (see `broadcasting <https://data-apis.org/array-api/latest\
        /API_specification/broadcasting.html>`_). Should have a floating-point data
        type.

    Returns
    -------
    out:
        an array containing the solution to the system ``AX = B`` for each square
        matrix. The returned array must have the same shape as ``x2`` (i.e., the array
        corresponding to ``B``) and must have a floating-point data type determined by
        `type-promotion <https://data-apis.org/array-api/latest/API_specification\
        /type_promotion.html>`_.
    """
    return _impl.solve(x1, x2)
