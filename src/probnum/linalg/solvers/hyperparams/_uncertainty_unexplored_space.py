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

    Phi: Optional[float] = None
    Psi: Optional[float] = None
