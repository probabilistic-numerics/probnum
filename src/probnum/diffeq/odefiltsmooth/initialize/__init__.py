"""Initialisation procedures for ODE filters."""

from ._initialize import InitializationRoutine
from ._initialize_with_taylormode import initialize_odefilter_with_taylormode
from ._runge_kutta import RungeKuttaInitialization

__all__ = [
    "InitializationRoutine",
    "RungeKuttaInitialization",
    "initialize_odefilter_with_taylormode",
]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
InitializationRoutine.__module__ = "probnum.diffeq.odefiltsmooth.initialize"
RungeKuttaInitialization.__module__ = "probnum.diffeq.odefiltsmooth.initialize"
