"""Initialization routines."""


from ._autodiff import ForwardMode, ForwardModeJVP, ReverseMode, TaylorMode
from ._odefilter_map import ODEFilterMAP
from ._scipy_fit import SciPyFit, SciPyFitWithJacobian
from ._stack import Stack, StackWithJacobian

# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
Stack.__module__ = "probnum.diffeq.odefilter.init"
StackWithJacobian.__module__ = "probnum.diffeq.init"
SciPyFit.__module__ = "probnum.diffeq.init"
SciPyFitWithJacobian.__module__ = "probnum.diffeq.init"
ForwardMode.__module__ = "probnum.diffeq.init"
ForwardModeJVP.__module__ = "probnum.diffeq.init"
ReverseMode.__module__ = "probnum.diffeq.init"
TaylorMode.__module__ = "probnum.diffeq.init"
ODEFilterMAP.__module__ = "probnum.diffeq.init"
