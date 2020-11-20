"""Differential Equations."""

from .ode import *
from .odefiltsmooth import *
from .odesolution import ODESolution
from .odesolver import *
from .steprule import *

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "ODE",
    "IVP",
    "logistic",
    "fitzhughnagumo",
    "lotkavolterra",
    "probsolve_ivp",
    "ODESolver",
    "GaussianIVPFilter",
    "ODEPrior",
    "IBM",
    "IOUP",
    "Matern",
    "ivp2ekf0",
    "ivp2ekf1",
    "ivp2ukf",
    "StepRule",
    "ConstantSteps",
    "AdaptiveSteps",
    "ODESolution",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODE.__module__ = "probnum.diffeq"
ODESolver.__module__ = "probnum.diffeq"
StepRule.__module__ = "probnum.diffeq"
