"""Exemplary state space model setups for Bayesian Filtering and Smoothing."""


from ._filtsmooth_problems import (
    benes_daum,
    car_tracking,
    logistic_ode,
    ornstein_uhlenbeck,
    pendulum,
)

__all__ = [
    "benes_daum",
    "car_tracking",
    "logistic_ode",
    "ornstein_uhlenbeck",
    "pendulum",
]
