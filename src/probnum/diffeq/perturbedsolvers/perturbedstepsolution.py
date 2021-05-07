from typing import Optional

import numpy as np
from scipy.integrate._ivp import rk

from probnum import _randomvariablelist, diffeq, randvars
from probnum.type import FloatArgType


class PerturbedStepSolution(diffeq.ODESolution):
    """Output of NoisyStepSolver."""

    def __init__(
        self,
        scales: list,
        times: np.ndarray,
        rvs: _randomvariablelist._RandomVariableList,
        interpolants: rk.RkDenseOutput,
    ):
        self.scales = scales
        self.interpolants = interpolants
        super().__init__(times, rvs)

    def interpolate(
        self,
        t: FloatArgType,
        previous_location: Optional[FloatArgType] = None,
        previous_state: Optional[randvars.RandomVariable] = None,
        next_location: Optional[FloatArgType] = None,
        next_state: Optional[randvars.RandomVariable] = None,
    ):
        discrete_location = list(self.locations).index(previous_location)

        # For the first and last state, no interpolation has to be performed.
        if t == self.locations[-1]:
            res = self.states[-1]
        if t == self.locations[0]:
            res = self.states[0]
        else:
            interpolant = self.interpolants[discrete_location]
            relative_time = (t - self.locations[discrete_location]) * self.scales[
                discrete_location
            ]
            res = randvars.Constant(
                interpolant(self.locations[discrete_location] + relative_time)
            )
        return res
