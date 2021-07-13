"""Linear-time implementation of optimization algorithms in state space models via
filtering and smoothing."""

from ._iterated_component import IteratedDiscreteComponent
from ._stoppingcriterion import StoppingCriterion

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "IteratedDiscreteComponent" "StoppingCriterion",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
StoppingCriterion.__module__ = "probnum.filtsmooth.optim"
IteratedDiscreteComponent.__module__ = "probnum.filtsmooth.optim"
