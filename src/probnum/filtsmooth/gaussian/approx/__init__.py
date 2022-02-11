"""Approximate Gaussian filtering and smoothing."""

from ._extendedkalman import ContinuousEKFComponent, DiscreteEKFComponent
from ._unscentedkalman import DiscreteUKFComponent

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ContinuousEKFComponent",
    "DiscreteEKFComponent",
    "DiscreteUKFComponent",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ContinuousEKFComponent.__module__ = "probnum.filtsmooth.gaussian.approx"
DiscreteEKFComponent.__module__ = "probnum.filtsmooth.gaussian.approx"
DiscreteUKFComponent.__module__ = "probnum.filtsmooth.gaussian.approx"
