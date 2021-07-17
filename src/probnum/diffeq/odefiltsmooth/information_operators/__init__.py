"""ODE Information operators."""

from ._information_operator import InformationOperator, ODEInformationOperator
from ._ode_residual import ExplicitODEResidual

__all__ = ["InformationOperator", "ODEInformationOperator", "ExplicitODEResidual"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
InformationOperator.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
ODEInformationOperator.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
ExplicitODEResidual.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
