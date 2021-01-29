"""Observation operators of probabilistic linear solvers."""

from ._matvec import MatVec, SampleMatVec
from ._observation_operator import ObservationOp

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ObservationOp", "MatVec", "SampleMatVec"]

# Set correct module paths. Corrects links and module paths in documentation.
ObservationOp.__module__ = "probnum.linalg.solvers.observation_ops"
MatVec.__module__ = "probnum.linalg.solvers.observation_ops"
SampleMatVec.__module__ = "probnum.linalg.solvers.observation_ops"
