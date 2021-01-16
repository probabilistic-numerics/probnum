"""Hyperparameter optimization routines for probabilistic linear solvers."""

from ._hyperparameter_optimization import HyperparameterOptimization
from ._optimal_noise_scale import OptimalNoiseScale
from ._uncertainty_calibration import UncertaintyCalibration

# Public classes and functions. Order is reflected in documentation.
__all__ = ["HyperparameterOptimization", "UncertaintyCalibration", "OptimalNoiseScale"]

# Set correct module paths. Corrects links and module paths in documentation.
HyperparameterOptimization.__module__ = "probnum.linalg.linearsolvers.hyperparam_optim"
UncertaintyCalibration.__module__ = "probnum.linalg.linearsolvers.hyperparam_optim"
OptimalNoiseScale.__module__ = "probnum.linalg.linearsolvers.hyperparam_optim"
