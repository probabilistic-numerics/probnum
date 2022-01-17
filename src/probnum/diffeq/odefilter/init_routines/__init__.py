"""Initialization routines."""

from ._autodiff import ForwardMode, ForwardModeJVP, ReverseMode, TaylorMode
from ._interface import InitializationRoutine
from ._non_probabilistic_fit import NonProbabilisticFit, NonProbabilisticFitWithJacobian
from ._odefilter_map import ODEFilterMAP
from ._stack import Stack, StackWithJacobian

__all__ = [
    "InitializationRoutine",
    "Stack",
    "StackWithJacobian",
    "NonProbabilisticFit",
    "NonProbabilisticFitWithJacobian",
    "ForwardMode",
    "ForwardModeJVP",
    "ReverseMode",
    "TaylorMode",
    "ODEFilterMAP",
]

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
InitializationRoutine.__module__ = "probnum.diffeq.odefilter.init_routines"
Stack.__module__ = "probnum.diffeq.odefilter.init_routines"
StackWithJacobian.__module__ = "probnum.diffeq.init_routines"
NonProbabilisticFit.__module__ = "probnum.diffeq.init_routines"
NonProbabilisticFitWithJacobian.__module__ = "probnum.diffeq.init_routines"
ForwardMode.__module__ = "probnum.diffeq.init_routines"
ForwardModeJVP.__module__ = "probnum.diffeq.init_routines"
ReverseMode.__module__ = "probnum.diffeq.init_routines"
TaylorMode.__module__ = "probnum.diffeq.init_routines"
ODEFilterMAP.__module__ = "probnum.diffeq.init_routines"
