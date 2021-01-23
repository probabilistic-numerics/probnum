"""Noise on the components of a linear system."""

import dataclasses
from typing import Optional

import probnum.random_variables as rvs

from ._hyperparameters import LinearSolverHyperparams


@dataclasses.dataclass(frozen=True)
class LinearSystemNoise(LinearSolverHyperparams):
    """Additive Gaussian noise on the system matrix and right hand side.

    Parameters
    ----------
    A_eps :
        Noise on the system matrix.
    b_eps :
        Noise on the right hand side.
    """

    A_eps: Optional[rvs.Normal] = None
    b_eps: Optional[rvs.Normal] = None
