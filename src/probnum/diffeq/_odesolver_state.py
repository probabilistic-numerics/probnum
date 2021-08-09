"""ODE Solver states."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from probnum import problems, randvars


@dataclass
class ODESolverState:
    """ODE solver states."""

    ivp: problems.InitialValueProblem

    t: float
    rv: randvars.RandomVariable
    error_estimate: Optional[np.ndarray] = None

    # The reference state is used for relative error estimation
    reference_state: Optional[np.ndarray] = None
