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
