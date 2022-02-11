"""Approximate Gaussian filtering and smoothing."""

from ._extendedkalman import ContinuousEKFComponent, DiscreteEKFComponent, EKFComponent
from ._unscentedkalman import ContinuousUKFComponent, DiscreteUKFComponent
from ._unscentedtransform import UnscentedTransform

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "EKFComponent",
    "ContinuousEKFComponent",
    "DiscreteEKFComponent",
    "ContinuousUKFComponent",
    "DiscreteUKFComponent",
    "UnscentedTransform",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
EKFComponent.__module__ = "probnum.filtsmooth.gaussian.approx"
ContinuousEKFComponent.__module__ = "probnum.filtsmooth.gaussian.approx"
DiscreteEKFComponent.__module__ = "probnum.filtsmooth.gaussian.approx"
ContinuousUKFComponent.__module__ = "probnum.filtsmooth.gaussian.approx"
DiscreteUKFComponent.__module__ = "probnum.filtsmooth.gaussian.approx"
UnscentedTransform.__module__ = "probnum.filtsmooth.gaussian.approx"
