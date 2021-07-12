"""Differential Equations.

This package defines probabilistic solvers for differential equations.
"""

from ._odesolution import ODESolution
from ._odesolver import ODESolver
from ._probsolve_ivp import probsolve_ivp

__all__ = ["_probsolve_ivp", "ODESolver", "ODESolution"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODESolver.__module__ = "probnum.diffeq"
ODESolution.__module__ = "probnum.diffeq"
