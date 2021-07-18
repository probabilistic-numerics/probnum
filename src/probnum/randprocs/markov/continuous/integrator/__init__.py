"""Transitions for integrated systems (e.g. integrated Wiener processes)."""

from ._ibm import IBM
from ._integrator import Integrator
from ._ioup import IOUP
from ._matern import Matern
from ._preconditioner import NordsieckLikeCoordinates, Preconditioner

__all__ = [
    "Integrator",
    "IBM",
    "IOUP",
    "Matern",
    "Preconditioner",
    "NordsieckLikeCoordinates",
]

# Set correct module paths. Corrects links and module paths in documentation.
Integrator.__module__ = "probnum.randprocs.markov.continuous.integrator"
IBM.__module__ = "probnum.randprocs.markov.continuous.integrator"
IOUP.__module__ = "probnum.randprocs.markov.continuous.integrator"
Matern.__module__ = "probnum.randprocs.markov.continuous.integrator"
Preconditioner.__module__ = "probnum.randprocs.markov.continuous.integrator"
NordsieckLikeCoordinates.__module__ = "probnum.randprocs.markov.continuous.integrator"
