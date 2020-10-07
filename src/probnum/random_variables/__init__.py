"""
Random variables are the main objects in ProbNum.

Random variables generalize multidimensional arrays by also representing uncertainty
about the (numerical) quantity in question. Despite their name, they do not have to
represent stochastic objects. Random variables are the primary in- and outputs of
probabilistic numerical methods.
"""

from ._dirac import Dirac
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
    "Dirac",
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

Dirac.__module__ = "probnum.random_variables"
Normal.__module__ = "probnum.random_variables"

asrandvar.__module__ = "probnum.random_variables"
