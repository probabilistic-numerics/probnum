"""Discrete-time transitions."""

from ._discrete_gaussian import (
    DiscreteGaussian,
    DiscreteLinearGaussian,
    DiscreteLTIGaussian,
)

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
