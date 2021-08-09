"""ODE Filtering."""

from . import utils
from ._odefilter import ODEFilter
from ._odefilter_solution import ODEFilterSolution

__all__ = ["ODEFilter", "ODEFilterSolution"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODEFilter.__module__ = "probnum.diffeq.odefilter"
ODEFilterSolution.__module__ = "probnum.diffeq.odefilter"
