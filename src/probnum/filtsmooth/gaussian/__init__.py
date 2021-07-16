"""Gaussian filtering and smoothing."""

from ._kalman import Kalman
from ._kalmanposterior import FilteringPosterior, KalmanPosterior, SmoothingPosterior

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Kalman",
    "KalmanPosterior",
    "FilteringPosterior",
    "SmoothingPosterior",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
Kalman.__module__ = "probnum.filtsmooth.gaussian"
KalmanPosterior.__module__ = "probnum.filtsmooth.gaussian"
FilteringPosterior.__module__ = "probnum.filtsmooth.gaussian"
SmoothingPosterior.__module__ = "probnum.filtsmooth.gaussian"
