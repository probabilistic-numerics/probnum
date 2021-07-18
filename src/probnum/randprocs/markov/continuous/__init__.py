"""Continous-time transitions and stochastic differential equations."""

from ._diffusions import ConstantDiffusion, Diffusion, PiecewiseConstantDiffusion
from ._sde import LTISDE, SDE, LinearSDE
from ._utils import matrix_fraction_decomposition

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "SDE",
    "LinearSDE",
    "LTISDE",
    "matrix_fraction_decomposition",
    "Diffusion",
    "ConstantDiffusion",
    "PiecewiseConstantDiffusion",
]

# Set correct module paths. Corrects links and module paths in documentation.
SDE.__module__ = "probnum.randprocs.markov.continuous"
LinearSDE.__module__ = "probnum.randprocs.markov.continuous"
LTISDE.__module__ = "probnum.randprocs.markov.continuous"
Diffusion.__module__ = "probnum.randprocs.markov.continuous"
ConstantDiffusion.__module__ = "probnum.randprocs.markov.continuous"
PiecewiseConstantDiffusion.__module__ = "probnum.randprocs.markov.continuous"
