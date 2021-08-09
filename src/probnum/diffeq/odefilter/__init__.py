"""ODE Filtering."""

from . import utils
from ._kalman_odesolution import KalmanODESolution
from ._odefilter import ODEFilter

__all__ = ["ODEFilter", "KalmanODESolution"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ODEFilter.__module__ = "probnum.diffeq.odefilter"
KalmanODESolution.__module__ = "probnum.diffeq.odefilter"
