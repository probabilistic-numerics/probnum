"""Utility functions for Bayesian filtering and smoothing."""

from ._merge_regression_problems import merge_regression_problems

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "merge_regression_problems",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
# via (e.g.) BayesFiltSmooth.__module__ = "probnum.filtsmooth.utils"
