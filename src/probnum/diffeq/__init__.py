"""Differential Equations.

This package defines common dynamical models and probabilistic solvers
for differential equations.
"""

from .ode import IVP, ODE, fitzhughnagumo, logistic, lorenz, lotkavolterra, seir
from .odefiltsmooth import (
    GaussianIVPFilter,
    KalmanODESolution,
    initialize_odefilter_with_rk,
    initialize_odefilter_with_taylormode,
    probsolve_ivp,
)
from .odesolution import ODESolution
from .odesolver import ODESolver
from .perturbedsolvers import (
    PerturbedStepSolution,
    PerturbedStepSolver,
    perturb_lognormal,
    perturb_uniform,
)
from .steprule import AdaptiveSteps, ConstantSteps, StepRule, propose_firststep
from .wrappedscipysolver import WrappedScipyRungeKutta

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
    "StepRule",
    "ConstantSteps",
    "AdaptiveSteps",
    "ODESolution",
    "KalmanODESolution",
    "propose_firststep",
    "initialize_odefilter_with_rk",
    "initialize_odefilter_with_taylormode",
    "PerturbedStepSolver",
    "PerturbedStepSolution",
    "WrappedScipyRungeKutta",
    "perturb_lognormal",
    "perturb_uniform",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODE.__module__ = "probnum.diffeq"
ODESolver.__module__ = "probnum.diffeq"
StepRule.__module__ = "probnum.diffeq"
