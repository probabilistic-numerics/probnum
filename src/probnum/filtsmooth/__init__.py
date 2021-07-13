"""Bayesian Filtering and Smoothing.

This package provides different kinds of Bayesian filters and smoothers
which estimate the distribution over observed and hidden variables in a
sequential model. The two operations differ by what information they
use. Filtering considers all observations up to a given point, while
smoothing takes the entire set of observations into account.
"""

from ._bayesfiltsmooth import BayesFiltSmooth
from ._timeseriesposterior import TimeSeriesPosterior

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "BayesFiltSmooth",
    "TimeSeriesPosterior",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
BayesFiltSmooth.__module__ = "probnum.filtsmooth"
TimeSeriesPosterior.__module__ = "probnum.filtsmooth"
