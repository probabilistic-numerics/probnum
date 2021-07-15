"""ODE Information operators."""

from ._information_operator import InformationOperator, ODEResidualOperator

__all__ = ["InformationOperator", "ODEResidualOperator"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
InformationOperator.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
ODEResidualOperator.__module__ = "probnum.diffeq.odefiltsmooth.information_operators"
