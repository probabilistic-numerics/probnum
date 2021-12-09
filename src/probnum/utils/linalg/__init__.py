"""Utility functions that involve numerical linear algebra."""

from ._cholesky_updates import cholesky_update, tril_to_positive_tril
from ._inner_product import euclidean_inprod, euclidean_norm
from ._orthogonalize import double_gram_schmidt, gram_schmidt, modified_gram_schmidt

__all__ = [
    "euclidean_inprod",
    "euclidean_norm",
    "cholesky_update",
    "tril_to_positive_tril",
    "gram_schmidt",
    "modified_gram_schmidt",
    "double_gram_schmidt",
]
