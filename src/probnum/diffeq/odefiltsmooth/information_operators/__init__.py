"""ODE Information operators."""

from ._approx_information_operator import (
    ApproximateInformationOperator,
    LocallyLinearizedInformationOperator,
)
from ._information_operator import InformationOperator, ODEInformationOperator
from ._ode_residual import ODEResidual

__all__ = [
    "InformationOperator",
    "ODEInformationOperator",
    "ODEResidual",
    "ApproximateInformationOperator",
    "LocallyLinearizedInformationOperator",
]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
InformationOperator.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
ODEInformationOperator.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
ODEResidual.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
ApproximateInformationOperator.__module__ = (
    "probnum.diffeq.odefiltsmooth.information_operators"
)
LocallyLinearizedInformationOperator.__module__ = (
    "probnum.diffeq.odefiltsmooth.information_operators"
)
