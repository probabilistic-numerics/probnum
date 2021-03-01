"""Differential Equations."""

from .ode import IVP, ODE, fitzhughnagumo, logistic, lorenz, lotkavolterra, seir
from .odefiltsmooth import (
    GaussianIVPFilter,
    KalmanODESolution,
    compute_all_derivatives_via_rk,
    compute_all_derivatives_via_taylormode,
    convert_coordwise_to_derivwise,
    convert_derivwise_to_coordwise,
    extend_ivp_with_all_derivatives,
    ivp2ekf0,
    ivp2ekf1,
    ivp2ukf,
    probsolve_ivp,
)
from .odesolution import ODESolution
from .odesolver import ODESolver
from .steprule import AdaptiveSteps, ConstantSteps, StepRule

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ODE",
    "IVP",
    "logistic",
    "fitzhughnagumo",
    "lotkavolterra",
    "seir",
    "lorenz",
    "probsolve_ivp",
    "ODESolver",
    "GaussianIVPFilter",
    "ivp2ekf0",
    "ivp2ekf1",
    "ivp2ukf",
    "StepRule",
    "ConstantSteps",
    "AdaptiveSteps",
    "ODESolution",
    "KalmanODESolution",
    "extend_ivp_with_all_derivatives",
    "compute_all_derivatives_via_rk",
    "compute_all_derivatives_via_taylormode",
    "convert_derivwise_to_coordwise",
    "convert_coordwise_to_derivwise",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODE.__module__ = "probnum.diffeq"
ODESolver.__module__ = "probnum.diffeq"
StepRule.__module__ = "probnum.diffeq"
