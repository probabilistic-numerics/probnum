"""Hyperparameters of probabilistic linear solvers."""

from ._hyperparameters import LinearSolverHyperparams
from ._noise import LinearSystemNoise
from ._uncertainty_unexplored_space import UncertaintyUnexploredSpace

# Public classes and functions. Order is reflected in documentation.
__all__ = ["LinearSolverHyperparams", "UncertaintyUnexploredSpace", "LinearSystemNoise"]

# Set correct module paths. Corrects links and module paths in documentation.
LinearSolverHyperparams.__module__ = "probnum.linalg.solvers.hyperparams"
UncertaintyUnexploredSpace.__module__ = "probnum.linalg.solvers.hyperparams"
LinearSystemNoise.__module__ = "probnum.linalg.solvers.hyperparams"
