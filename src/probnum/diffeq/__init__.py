"""Differential Equations.

This package defines probabilistic solvers for differential equations.
"""

# Publicly visible subpackage: "perturbed" needs to be imported here
# Without this line, a double-subpackage import does not work?!
from . import perturbed
from ._odesolution import ODESolution
from ._odesolver import ODESolver
from ._probsolve_ivp import probsolve_ivp

__all__ = ["_probsolve_ivp", "ODESolver", "ODESolution"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODESolver.__module__ = "probnum.diffeq"
ODESolution.__module__ = "probnum.diffeq"
