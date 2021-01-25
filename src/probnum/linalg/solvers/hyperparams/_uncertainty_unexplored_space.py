"""Uncertainty in the unexplored space of a linear solver."""

import dataclasses
from typing import Optional

from ._hyperparameters import LinearSolverHyperparams


@dataclasses.dataclass(frozen=True)
class UncertaintyUnexploredSpace(LinearSolverHyperparams):
    r"""Uncertainty scales of the system matrix and inverse model.

    Parameters
    ----------
    Phi :
        Uncertainty scaling :math:`\Phi` of the belief about the matrix in the unexplored
        action space :math:`\operatorname{span}(s_1, \dots, s_k)^\perp`.
    Psi :
        Uncertainty scaling :math:`\Psi` of the belief about the inverse in the
        unexplored observation space :math:`\operatorname{span}(y_1, \dots, y_k)^\perp`.
    """

    Phi: float = 1.0
    Psi: float = 1.0

    def __post_init__(self):
        if self.Phi < 0.0 or self.Psi < 0.0:
            raise ValueError(
                "Uncertainty scales must be non-negative, but are "
                f"Phi={self.Phi} and Psi={self.Psi}"
            )
