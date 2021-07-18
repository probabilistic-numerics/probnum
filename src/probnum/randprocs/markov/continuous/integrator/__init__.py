"""Transitions for integrated systems (e.g. integrated Wiener processes)."""

from . import utils
from ._ibm import IntegratedWienerProcess, IntegratedWienerProcessTransition
from ._integrator import IntegratorTransition
from ._ioup import (
    IntegratedOrnsteinUhlenbeckProcess,
    IntegratedOrnsteinUhlenbeckProcessTransition,
)
from ._matern import Matern
from ._preconditioner import NordsieckLikeCoordinates, Preconditioner

__all__ = [
    "IntegratorTransition",
    "IntegratedWienerProcess",
    "IntegratedWienerProcessTransition",
    "IntegratedOrnsteinUhlenbeckProcess",
    "IntegratedOrnsteinUhlenbeckProcessTransition",
    "Matern",
    "Preconditioner",
    "NordsieckLikeCoordinates",
]

# Set correct module paths. Corrects links and module paths in documentation.
IntegratorTransition.__module__ = "probnum.randprocs.markov.continuous.integrator"
IntegratedWienerProcess.__module__ = "probnum.randprocs.markov.continuous.integrator"
IntegratedWienerProcessTransition.__module__ = (
    "probnum.randprocs.markov.continuous.integrator"
)
IntegratedOrnsteinUhlenbeckProcess.__module__ = (
    "probnum.randprocs.markov.continuous.integrator"
)
IntegratedOrnsteinUhlenbeckProcessTransition.__module__ = (
    "probnum.randprocs.markov.continuous.integrator"
)
Matern.__module__ = "probnum.randprocs.markov.continuous.integrator"
Preconditioner.__module__ = "probnum.randprocs.markov.continuous.integrator"
NordsieckLikeCoordinates.__module__ = "probnum.randprocs.markov.continuous.integrator"
