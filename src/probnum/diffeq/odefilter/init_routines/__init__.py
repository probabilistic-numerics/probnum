"""Initialization routines for ODE filters.

To choose, you may use the following rough guidelines.
But feel invited to play around with the options and be creative.

* Order 1: Stack()
* Order 2: StackWithJacobian() if a Jacobian is available, or NonProbabilisticFit() if not.
  If Jax is available, compute the Jacobian and use StackWithJacobian(),
  (or choose ForwardModeJVP() altogether).
* Order 3, 4, 5: NonProbabilisticFitWithJacobian() if the Jacobian of the ODE vector field
  is available, or NonProbabilisticFit() if not.
* Order >5: TaylorMode(). For orders 6 and 7, ForwardModeJVP() might work well too.
"""

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
