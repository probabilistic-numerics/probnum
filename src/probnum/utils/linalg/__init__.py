"""Utility functions that involve numerical linear algebra."""

from ._cholesky_updates import cholesky_update, tril_to_positive_tril

__all__ = ["cholesky_update", "tril_to_positive_tril"]
