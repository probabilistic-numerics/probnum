"""Bayesian quadrature methods and their components."""

from . import (
    acquisition_functions,
    belief_updates,
    initial_designs,
    policies,
    stopping_criteria,
)
from ._bayesian_quadrature import BayesianQuadrature
from ._bq_state import BQIterInfo, BQState

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "BayesianQuadrature",
    "BQState",
    "BQIterInfo",
]

# Set correct module paths. Corrects links and module paths in documentation.
BayesianQuadrature.__module__ = "probnum.quad.solvers"
BQState.__module__ = "probnum.quad.solvers"
BQIterInfo.__module__ = "probnum.quad.solvers"
