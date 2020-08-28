"""
This package implements random variables. Random variables are the primary in- and
outputs of probabilistic numerical methods. A generic signature of such methods looks
like this:

.. highlight:: python
.. code-block:: python

    randvar_out, info = probnum_method(problem, randvar_in, **kwargs)

"""

from ._random_variable import (
    RandomVariable,
    DiscreteRandomVariable,
    ContinuousRandomVariable,
)

from ._dirac import Dirac
from ._normal import Normal

from ._scipy_stats import (
    WrappedSciPyRandomVariable,
    WrappedSciPyDiscreteRandomVariable,
    WrappedSciPyContinuousRandomVariable,
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
