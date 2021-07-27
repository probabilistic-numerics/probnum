"""ODE Filtering."""

from . import utils
from ._ivpfiltsmooth import GaussianIVPFilter
from ._kalman_odesolution import KalmanODESolution

__all__ = ["GaussianIVPFilter", "KalmanODESolution"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
GaussianIVPFilter.__module__ = "probnum.diffeq.odefiltsmooth"
KalmanODESolution.__module__ = "probnum.diffeq.odefiltsmooth"
