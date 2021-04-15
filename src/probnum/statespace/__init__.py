"""Probabilistic State Space Models.

This package implements continuous-discrete and discrete-discrete state
space models, which are the basis for Bayesian filtering and smoothing,
but also probabilistic ODE solvers.
"""

from .discrete_transition import (
    DiscreteGaussian,
    DiscreteLinearGaussian,
    DiscreteLTIGaussian,
)
from .discrete_transition_utils import (
    condition_state_on_measurement,
    condition_state_on_rv,
)
from .generate_samples import generate_samples
from .integrator import IBM, IOUP, Integrator, Matern
from .preconditioner import NordsieckLikeCoordinates, Preconditioner
from .sde import LTISDE, SDE, LinearSDE
from .sde_utils import matrix_fraction_decomposition
from .transition import Transition

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Transition",
    "SDE",
    "LinearSDE",
    "LTISDE",
    "Integrator",
    "IBM",
    "IOUP",
    "Matern",
    "DiscreteGaussian",
    "DiscreteLinearGaussian",
    "DiscreteLTIGaussian",
    "Preconditioner",
    "NordsieckLikeCoordinates",
    "matrix_fraction_decomposition",
    "generate_samples",
    "condition_state_on_measurement",
    "condition_state_on_rv",
]

# Set correct module paths. Corrects links and module paths in documentation.
Transition.__module__ = "probnum.statespace"
SDE.__module__ = "probnum.statespace"
LinearSDE.__module__ = "probnum.statespace"
LTISDE.__module__ = "probnum.statespace"
Integrator.__module__ = "probnum.statespace"
IBM.__module__ = "probnum.statespace"
IOUP.__module__ = "probnum.statespace"
Matern.__module__ = "probnum.statespace"
DiscreteGaussian.__module__ = "probnum.statespace"
DiscreteLinearGaussian.__module__ = "probnum.statespace"
DiscreteLTIGaussian.__module__ = "probnum.statespace"
Preconditioner.__module__ = "probnum.statespace"
NordsieckLikeCoordinates.__module__ = "probnum.statespace"
