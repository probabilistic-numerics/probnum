"""Differential Equations."""

from .ode import (
    IVP,
    ODE,
    fitzhughnagumo,
    logistic,
    lotkavolterra,
    rigidbody,
    seir,
    threebody,
    vanderpol,
)
from .odefiltsmooth import GaussianIVPFilter, ivp2ekf0, ivp2ekf1, ivp2ukf, probsolve_ivp
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
    "rigidbody",
    "vanderpol",
    "threebody",
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
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODE.__module__ = "probnum.diffeq"
ODESolver.__module__ = "probnum.diffeq"
StepRule.__module__ = "probnum.diffeq"
