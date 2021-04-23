"""Exemplary state space model setups for Bayesian Filtering and Smoothing."""


from ._filtsmooth_problems import (
    car_tracking,
    logistic_ode,
    ornstein_uhlenbeck,
    pendulum,
)

__all__ = ["car_tracking", "logistic_ode", "ornstein_uhlenbeck", "pendulum"]
