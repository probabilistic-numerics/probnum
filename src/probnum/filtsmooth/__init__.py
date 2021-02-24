"""Bayesian Filtering and Smoothing.

This package provides different kinds of Bayesian filters and smoothers
which estimate the distribution over observed and hidden variables in a
sequential model. The two operations differ by what information they
use. Filtering considers all observations up to a given point, while
smoothing takes the entire set of observations into account.
"""

from .bayesfiltsmooth import BayesFiltSmooth
from .filtsmoothposterior import FiltSmoothPosterior
from .gaussfiltsmooth import (
    ContinuousEKFComponent,
    ContinuousUKFComponent,
    DiscreteEKFComponent,
    DiscreteUKFComponent,
    EKFComponent,
    FilteringPosterior,
    IteratedDiscreteComponent,
    Kalman,
    KalmanPosterior,
    SmoothingPosterior,
    StoppingCriterion,
    UKFComponent,
    UnscentedTransform,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "BayesFiltSmooth",
    "Kalman",
    "EKFComponent",
    "ContinuousEKFComponent",
    "DiscreteEKFComponent",
    "UKFComponent",
    "ContinuousUKFComponent",
    "DiscreteUKFComponent",
    "UnscentedTransform",
    "FiltSmoothPosterior",
    "KalmanPosterior",
    "FilteringPosterior",
    "SmoothingPosterior",
    "StoppingCriterion",
    "IteratedDiscreteComponent",
]
