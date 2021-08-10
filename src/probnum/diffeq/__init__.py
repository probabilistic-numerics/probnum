"""Differential Equations.

This package implements probabilistic solvers for differential equations.
"""

# Publicly visible subpackage "perturbed" needs to be imported here
# Without this line, a double-subpackage import as in
#     import diffeq
#     solver = diffeq.perturbed.step.PerturbedStepSolver(...)
# does not work.
from . import perturbed
from ._odesolution import ODESolution
from ._odesolver import ODESolver
from ._odesolver_state import ODESolverState
from ._perturbsolve_ivp import perturbsolve_ivp
from ._probsolve_ivp import probsolve_ivp

__all__ = [
    "probsolve_ivp",
    "perturbsolve_ivp",
    "ODESolver",
    "ODESolution",
    "ODESolverState",
]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODESolver.__module__ = "probnum.diffeq"
ODESolution.__module__ = "probnum.diffeq"
ODESolverState.__module__ = "probnum.diffeq"
