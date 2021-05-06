"""Random Variables.

Random variables generalize multi-dimensional arrays by encoding
uncertainty about the (numerical) quantity in question. Despite their
name, they do not necessarily represent stochastic objects. Random
variables are also the primary in- and outputs of probabilistic
numerical methods.
"""

from ._categorical import Categorical
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
    "Categorical",
    "WrappedSciPyRandomVariable",
    "WrappedSciPyDiscreteRandomVariable",
    "WrappedSciPyContinuousRandomVariable",
]

# Set correct module paths. Corrects links and module paths in documentation.
RandomVariable.__module__ = "probnum.randvars"
DiscreteRandomVariable.__module__ = "probnum.randvars"
ContinuousRandomVariable.__module__ = "probnum.randvars"

WrappedSciPyRandomVariable.__module__ = "probnum.randvars"
WrappedSciPyDiscreteRandomVariable.__module__ = "probnum.randvars"
WrappedSciPyContinuousRandomVariable.__module__ = "probnum.randvars"

Constant.__module__ = "probnum.randvars"
Normal.__module__ = "probnum.randvars"
Categorical.__module__ = "probnum.randvars"

asrandvar.__module__ = "probnum.randvars"
