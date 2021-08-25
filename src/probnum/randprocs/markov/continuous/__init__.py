"""Continous-time transitions and stochastic differential equations."""

from ._diffusions import ConstantDiffusion, Diffusion, PiecewiseConstantDiffusion
from ._linear_sde import LinearSDE
from ._lti_sde import LTISDE
from ._mfd import matrix_fraction_decomposition
from ._sde import SDE

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "SDE",
    "LinearSDE",
    "LTISDE",
    "Diffusion",
    "ConstantDiffusion",
    "PiecewiseConstantDiffusion",
    "matrix_fraction_decomposition",
]

# Set correct module paths. Corrects links and module paths in documentation.
SDE.__module__ = "probnum.randprocs.markov.continuous"
LinearSDE.__module__ = "probnum.randprocs.markov.continuous"
LTISDE.__module__ = "probnum.randprocs.markov.continuous"
Diffusion.__module__ = "probnum.randprocs.markov.continuous"
ConstantDiffusion.__module__ = "probnum.randprocs.markov.continuous"
PiecewiseConstantDiffusion.__module__ = "probnum.randprocs.markov.continuous"
matrix_fraction_decomposition.__module__ = "probnum.randprocs.markov.continuous"
