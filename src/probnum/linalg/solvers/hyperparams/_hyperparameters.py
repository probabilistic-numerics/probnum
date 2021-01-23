"""Hyperparameters of probabilistic linear solvers."""

import dataclasses
from abc import ABC


@dataclasses.dataclass(frozen=True)
class LinearSolverHyperparams(ABC):
    """Hyperparameters of a belief over quantities of interest of a linear system."""

    pass
