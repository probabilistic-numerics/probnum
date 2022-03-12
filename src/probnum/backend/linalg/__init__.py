"""Backend functions for linear algebra."""

__all__ = [
    "norm",
    "induced_norm",
    "inner_product",
    "gram_schmidt",
    "modified_gram_schmidt",
    "double_gram_schmidt",
    "cholesky",
    "solve_triangular",
    "solve_cholesky",
    "cholesky_update",
    "tril_to_positive_tril",
    "qr",
    "svd",
]

from .. import BACKEND, Backend

if BACKEND is Backend.NUMPY:
    from ._numpy import *
elif BACKEND is Backend.JAX:
    from ._jax import *
elif BACKEND is Backend.TORCH:
    from ._torch import *

from ._cholesky_updates import cholesky_update, tril_to_positive_tril
from ._inner_product import induced_norm, inner_product
from ._orthogonalize import double_gram_schmidt, gram_schmidt, modified_gram_schmidt
