"""Discrete-time transitions."""

from ._condition_state import condition_state_on_measurement, condition_state_on_rv
from ._discrete_gaussian import (
    DiscreteGaussian,
    DiscreteLinearGaussian,
    DiscreteLTIGaussian,
)

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "DiscreteGaussian",
    "DiscreteLinearGaussian",
    "DiscreteLTIGaussian",
    "condition_state_on_rv",
    "condition_state_on_measurement",
]

# Set correct module paths. Corrects links and module paths in documentation.
DiscreteGaussian.__module__ = "probnum.randprocs.markov.discrete"
DiscreteLinearGaussian.__module__ = "probnum.randprocs.markov.discrete"
DiscreteLTIGaussian.__module__ = "probnum.randprocs.markov.discrete"
