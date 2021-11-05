"""Abstract base classes for (components of) probabilistic numerical methods."""

from ._probabilistic_numerical_method import ProbabilisticNumericalMethod
from ._stopping_criterion import StoppingCriterion

# Public classes and functions. Order is reflected in documentation.
__all__ = ["ProbabilisticNumericalMethod", "StoppingCriterion"]

# Set correct module paths. Corrects links and module paths in documentation.
ProbabilisticNumericalMethod.__module__ = "probnum.pnm"
StoppingCriterion.__module__ = "probnum.pnm"
