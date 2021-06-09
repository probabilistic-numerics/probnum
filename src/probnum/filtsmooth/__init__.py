"""Bayesian Filtering and Smoothing.

This package provides different kinds of Bayesian filters and smoothers
which estimate the distribution over observed and hidden variables in a
sequential model. The two operations differ by what information they
use. Filtering considers all observations up to a given point, while
smoothing takes the entire set of observations into account.
"""

from ._bayesfiltsmooth import BayesFiltSmooth
from ._gaussfiltsmooth import (
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
from ._particlefiltsmooth import (
    BootstrapImportanceDistribution,
    ImportanceDistribution,
    LinearizationImportanceDistribution,
    ParticleFilter,
    ParticleFilterPosterior,
    effective_number_of_events,
)
from ._timeseriesposterior import TimeSeriesPosterior
from ._utils import merge_regression_problems

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
    "TimeSeriesPosterior",
    "KalmanPosterior",
    "FilteringPosterior",
    "SmoothingPosterior",
    "StoppingCriterion",
    "IteratedDiscreteComponent",
    "ParticleFilter",
    "ParticleFilterPosterior",
    "effective_number_of_events",
    "merge_regression_problems",
    "ImportanceDistribution",
    "BootstrapImportanceDistribution",
    "LinearizationImportanceDistribution",
]
