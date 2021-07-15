"""Fast optimization algorithms in state space models via filtering and smoothing."""

from ._gauss_newton import GaussNewton
from ._iterated_component import IteratedDiscreteComponent
from ._state_space_optimizer import StateSpaceOptimizer
from ._stoppingcriterion import StoppingCriterion

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "StateSpaceOptimizer",
    "GaussNewton",
    "IteratedDiscreteComponent",
    "StoppingCriterion",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
StateSpaceOptimizer.__module__ = "probnum.filtsmooth.optim"
GaussNewton.__module__ = "probnum.filtsmooth.optim"
StoppingCriterion.__module__ = "probnum.filtsmooth.optim"
IteratedDiscreteComponent.__module__ = "probnum.filtsmooth.optim"
