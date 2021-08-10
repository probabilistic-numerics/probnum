"""Event handling in ProbNum ODE solvers."""

from ._callback import ODESolverCallback
from ._discrete_callback import DiscreteCallback

__all__ = [
    "ODESolverCallback",
    "DiscreteCallback",
]


ODESolverCallback.__module__ = "probnum.diffeq.callbacks"
DiscreteCallback.__module__ = "probnum.diffeq.callbacks"
