"""Initialization routines."""

from ._autodiff import ForwardMode, ForwardModeJVP, ReverseMode, TaylorMode
from ._interface import InitializationRoutine
from ._odefilter_map import ODEFilterMAP
from ._scipy_fit import SciPyFit, SciPyFitWithJacobian
from ._stack import Stack, StackWithJacobian

__all__ = [
    "InitializationRoutine",
    "Stack",
    "StackWithJacobian",
    "SciPyFit",
    "SciPyFitWithJacobian",
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
SciPyFit.__module__ = "probnum.diffeq.init_routines"
SciPyFitWithJacobian.__module__ = "probnum.diffeq.init_routines"
ForwardMode.__module__ = "probnum.diffeq.init_routines"
ForwardModeJVP.__module__ = "probnum.diffeq.init_routines"
ReverseMode.__module__ = "probnum.diffeq.init_routines"
TaylorMode.__module__ = "probnum.diffeq.init_routines"
ODEFilterMAP.__module__ = "probnum.diffeq.init_routines"
