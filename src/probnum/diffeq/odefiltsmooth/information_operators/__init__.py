"""ODE Information operators."""

from ._information_operator import InformationOperator, ODEInformationOperator
from ._ode_residual import ODEResidual

__all__ = ["InformationOperator", "ODEInformationOperator", "ODEResidual"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
InformationOperator.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
ODEInformationOperator.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
ODEResidual.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
