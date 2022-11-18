"""Kernel embeddings for Bayesian quadrature methods."""

from ._kernel_embedding import KernelEmbedding

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "KernelEmbedding",
]

# Set correct module paths. Corrects links and module paths in documentation.
KernelEmbedding.__module__ = "probnum.quad.kernel_embeddings"
