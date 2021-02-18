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
    Kalman,
    KalmanPosterior,
    StoppingCriterion,
    UKFComponent,
    UnscentedTransform,
    cholesky_update,
    condition_state_on_measurement,
    iterate_update,
    linear_system_matrices,
    measure_sqrt,
    measure_via_transition,
    predict_sqrt,
    predict_via_transition,
    rts_add_precon,
    rts_smooth_step_classic,
    rts_smooth_step_sqrt,
    triu_to_positive_tril,
    update_classic,
    update_sqrt,
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
    "StoppingCriterion",
    "cholesky_update",
    "triu_to_positive_tril",
    "predict_via_transition",
    "predict_sqrt",
    "measure_via_transition",
    "measure_sqrt",
    "update_classic",
    "update_sqrt",
    "condition_state_on_measurement",
    "iterate_update",
    "rts_add_precon",
    "rts_smooth_step_classic",
    "rts_smooth_step_sqrt",
]
