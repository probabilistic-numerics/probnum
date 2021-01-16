"""Policies of probabilistic linear solvers returning actions."""

from ._conjugate_directions import ConjugateDirections
from ._explore_exploit import ExploreExploit
from ._policy import Policy
from ._thompson_sampling import ThompsonSampling

# Public classes and functions. Order is reflected in documentation.
__all__ = ["Policy", "ConjugateDirections", "ThompsonSampling", "ExploreExploit"]

# Set correct module paths. Corrects links and module paths in documentation.
Policy.__module__ = "probnum.linalg.linearsolvers.policies"
ConjugateDirections.__module__ = "probnum.linalg.linearsolvers.policies"
ThompsonSampling.__module__ = "probnum.linalg.linearsolvers.policies"
ExploreExploit.__module__ = "probnum.linalg.linearsolvers.policies"
