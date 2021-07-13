"""Scipy wrappers into the ODESolver and ODESolution interfaces."""

from ._wrapped_scipy_odesolution import WrappedScipyODESolution
from ._wrapped_scipy_solver import WrappedScipyRungeKutta

__all__ = ["WrappedScipyRungeKutta", "WrappedScipyODESolution"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
WrappedScipyRungeKutta.__module__ = "probnum.diffeq.perturbed.scipy_wrapper"
WrappedScipyODESolution.__module__ = "probnum.diffeq.perturbed.scipy_wrapper"
