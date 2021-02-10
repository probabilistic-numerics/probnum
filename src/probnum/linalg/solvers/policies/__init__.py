"""Policies of probabilistic linear solvers returning actions."""

from ._conjugate_directions import ConjugateDirections
from ._explore_exploit import ExploreExploit
from ._max_supnorm_column import MaxSupNormColumn
from ._policy import Policy
from ._residual import Residual
from ._thompson_sampling import ThompsonSampling

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Policy",
    "Residual",
    "ConjugateDirections",
    "ThompsonSampling",
    "ExploreExploit",
    "MaxSupNormColumn",
]

# Set correct module paths. Corrects links and module paths in documentation.
Policy.__module__ = "probnum.linalg.solvers.policies"
Residual.__module__ = "probnum.linalg.solvers.policies"
ConjugateDirections.__module__ = "probnum.linalg.solvers.policies"
ThompsonSampling.__module__ = "probnum.linalg.solvers.policies"
ExploreExploit.__module__ = "probnum.linalg.solvers.policies"
MaxSupNormColumn.__module__ = "probnum.linalg.solvers.policies"
