"""Utility functions that involve numerical linear algebra."""

from ._cholesky_updates import cholesky_update, tril_to_positive_tril
from ._inner_product import induced_norm, inner_product
from ._orthogonalize import double_gram_schmidt, gram_schmidt, modified_gram_schmidt

__all__ = [
    "inner_product",
    "induced_norm",
    "cholesky_update",
    "tril_to_positive_tril",
    "gram_schmidt",
    "modified_gram_schmidt",
    "double_gram_schmidt",
]
