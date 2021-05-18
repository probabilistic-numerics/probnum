from typing import List, Optional

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
        interpolants: List[rk.DenseOutput],
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
        # Find the index of the previous location. This is needed to access the correct
        # interpolant as the interpolation in SciPy is a concatenation of dense outputs.
        discrete_location = list(self.locations).index(previous_location)

        # For the first state, no interpolation has to be performed.
        if t == self.locations[0]:
            res = self.states[0]
        else:
            interpolant = self.interpolants[discrete_location]
            relative_time = (t - previous_location) * self.scales[discrete_location]
            res = randvars.Constant(interpolant(previous_location + relative_time))
        return res
