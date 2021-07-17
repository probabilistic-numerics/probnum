"""Initialisation procedures for ODE filters."""

from ._initialize import InitializationRoutine
from ._runge_kutta import RungeKuttaInitialization
from ._taylor_mode import TaylorModeInitialization

__all__ = [
    "InitializationRoutine",
    "RungeKuttaInitialization",
    "TaylorModeInitialization",
]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
InitializationRoutine.__module__ = "probnum.diffeq.odefiltsmooth.initialize"
RungeKuttaInitialization.__module__ = "probnum.diffeq.odefiltsmooth.initialize"
TaylorModeInitialization.__module__ = "probnum.diffeq.odefiltsmooth.initialize"
