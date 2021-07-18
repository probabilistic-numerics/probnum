"""Transitions for integrated systems (e.g. integrated Wiener processes)."""

from . import utils
from ._integrator import IntegratorTransition
from ._ioup import (
    IntegratedOrnsteinUhlenbeckProcess,
    IntegratedOrnsteinUhlenbeckTransition,
)
from ._iwp import IntegratedWienerProcess, IntegratedWienerTransition
from ._matern import MaternProcess, MaternTransition
from ._preconditioner import NordsieckLikeCoordinates, Preconditioner

__all__ = [
    "IntegratorTransition",
    "IntegratedWienerProcess",
    "IntegratedWienerTransition",
    "IntegratedOrnsteinUhlenbeckProcess",
    "IntegratedOrnsteinUhlenbeckTransition",
    "MaternProcess",
    "MaternTransition",
    "Preconditioner",
    "NordsieckLikeCoordinates",
]

# Set correct module paths. Corrects links and module paths in documentation.
IntegratorTransition.__module__ = "probnum.randprocs.markov.continuous.integrator"
IntegratedWienerProcess.__module__ = "probnum.randprocs.markov.continuous.integrator"
IntegratedWienerTransition.__module__ = "probnum.randprocs.markov.continuous.integrator"
IntegratedOrnsteinUhlenbeckProcess.__module__ = (
    "probnum.randprocs.markov.continuous.integrator"
)
IntegratedOrnsteinUhlenbeckTransition.__module__ = (
    "probnum.randprocs.markov.continuous.integrator"
)
MaternProcess.__module__ = "probnum.randprocs.markov.continuous.integrator"
MaternTransition.__module__ = "probnum.randprocs.markov.continuous.integrator"
Preconditioner.__module__ = "probnum.randprocs.markov.continuous.integrator"
NordsieckLikeCoordinates.__module__ = "probnum.randprocs.markov.continuous.integrator"
