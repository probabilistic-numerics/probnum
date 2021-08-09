"""ODE Filtering."""

from . import utils
from ._odefilter import ODEFilter
from ._odefilter_solution import KalmanODESolution

__all__ = ["ODEFilter", "KalmanODESolution"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODEFilter.__module__ = "probnum.diffeq.odefilter"
KalmanODESolution.__module__ = "probnum.diffeq.odefilter"
