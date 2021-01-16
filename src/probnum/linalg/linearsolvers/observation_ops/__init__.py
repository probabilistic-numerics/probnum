"""Observation operators of probabilistic linear solvers."""

from ._matvec import MatVecObservation
from ._observation_operator import ObservationOperator

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ObservationOperator", "MatVecObservation"]

# Set correct module paths. Corrects links and module paths in documentation.
ObservationOperator.__module__ = "probnum.linalg.linearsolvers.observation_ops"
MatVecObservation.__module__ = "probnum.linalg.linearsolvers.observation_ops"
