"""Random Variables."""

from ._constant import Constant
from ._normal import Normal
from ._random_variable import (
    ContinuousRandomVariable,
    DiscreteRandomVariable,
    RandomVariable,
)
from ._scipy_stats import (
    WrappedSciPyContinuousRandomVariable,
    WrappedSciPyDiscreteRandomVariable,
    WrappedSciPyRandomVariable,
)
from ._utils import asrandvar

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "asrandvar",
    "RandomVariable",
    "DiscreteRandomVariable",
    "ContinuousRandomVariable",
    "Constant",
    "Normal",
    "WrappedSciPyRandomVariable",
    "WrappedSciPyDiscreteRandomVariable",
    "WrappedSciPyContinuousRandomVariable",
]

# Set correct module paths. Corrects links and module paths in documentation.
RandomVariable.__module__ = "probnum.random_variables"
DiscreteRandomVariable.__module__ = "probnum.random_variables"
ContinuousRandomVariable.__module__ = "probnum.random_variables"

WrappedSciPyRandomVariable.__module__ = "probnum.random_variables"
WrappedSciPyDiscreteRandomVariable.__module__ = "probnum.random_variables"
WrappedSciPyContinuousRandomVariable.__module__ = "probnum.random_variables"

Constant.__module__ = "probnum.random_variables"
Normal.__module__ = "probnum.random_variables"

asrandvar.__module__ = "probnum.random_variables"
