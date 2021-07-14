"""Output of PerturbedStepSolver."""

from typing import List, Optional

import numpy as np
from scipy.integrate._ivp import rk

from probnum import _randomvariablelist, randvars
from probnum.diffeq import _odesolution
from probnum.typing import FloatArgType


class PerturbedStepSolution(_odesolution.ODESolution):
    """Probabilistic ODE solution corresponding to the :class:`PerturbedStepSolver`."""

    def __init__(
        self,
        scales: List[float],
        locations: np.ndarray,
        states: _randomvariablelist._RandomVariableList,
        interpolants: List[rk.DenseOutput],
    ):
        self.scales = scales
        self.interpolants = interpolants
        super().__init__(locations, states)

    def interpolate(
        self,
        t: FloatArgType,
        previous_index: Optional[FloatArgType] = None,
        next_index: Optional[FloatArgType] = None,
    ):
        # For the first state, no interpolation has to be performed.
        if t == self.locations[0]:
            return self.states[0]
        if t == self.locations[-1]:
            return self.states[-1]
        else:
            interpolant = self.interpolants[previous_index]
            relative_time = (t - self.locations[previous_index]) * self.scales[
                previous_index
            ]
            previous_time = self.locations[previous_index]
            evaluation = interpolant(previous_time + relative_time)
            res_as_rv = randvars.Constant(evaluation)
        return res_as_rv
