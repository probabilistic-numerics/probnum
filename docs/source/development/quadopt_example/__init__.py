"""
Example Implementation of 1D (noisy) quadratic optimization.
"""

from ._quadratic_programming import probsolve_qp, ProbabilisticQuadraticOptimizer
from .policies import QuadOptPolicy
from .observation_operators import QuadOptObservation
from .belief_updates import QuadOptBeliefUpdate
from .stopping_criteria import QuadOptStoppingCriterion

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "probsolve_qp",
    "ProbabilisticQuadraticOptimizer",
    "QuadOptPolicy",
    "QuadOptObservation",
    "QuadOptStoppingCriterion",
]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticQuadraticOptimizer.__module__ = "probnum.quadopt_example"
QuadOptPolicy.__module__ = "probnum.quadopt_example"
QuadOptObservation.__module__ = "probnum.quadopt_example"
QuadOptStoppingCriterion.__module__ = "probnum.quadopt_example"
