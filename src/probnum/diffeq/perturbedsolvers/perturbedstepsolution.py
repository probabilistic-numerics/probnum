import numpy as np
from scipy.integrate._ivp import rk

from probnum import _randomvariablelist, diffeq, randvars
from probnum.filtsmooth.timeseriesposterior import (
    DenseOutputLocationArgType,
    DenseOutputValueType,
)


class PerturbedStepSolution(diffeq.ODESolution):
    """Output of NoisyStepSolver."""

    def __init__(
        self,
        scales: np.array,
        times: np.array,
        rvs: np.array,
        interpolants: rk.RkDenseOutput,
    ):
        self.scales = scales
        self.interpolants = interpolants
        super().__init__(times, rvs)

    def __call__(self, t: DenseOutputLocationArgType) -> DenseOutputValueType:
        if not np.isscalar(t):

            # recursive evaluation (t can now be any array, not just length 1!)
            return _randomvariablelist._RandomVariableList(
                [self.__call__(t_pt) for t_pt in np.asarray(t)]
            )

        # Access last state directly.
        if t == self.locations[-1]:
            res = self.states[-1]
        else:

            # get index of closest left interpolant (í.e. correct interpolant).
            closest_left_t = get_interpolant(self.locations, t)
            interpolant = self.interpolants[closest_left_t]
            if self.locations[closest_left_t] == t:
                res = randvars.Constant(interpolant(t))
            else:
                relative_time = (t - self.locations[closest_left_t]) * self.scales[
                    closest_left_t
                ]
                res = randvars.Constant(
                    interpolant(self.locations[closest_left_t] + relative_time)
                )
        return res


def get_interpolant(times: np.array, t_new: float):
    """in a sorted array times find the element t that is the closest before t_new
    Parameters
    ----------
    times : array
        array of discrete evaluation points [t_1, t_2, ...t_n]
    t_new : float
        timepoint t_new of which we want to find the closest left point in
        times
    Returns
    -------
    closest left timepoint of t in times. Returns index of t if t is in times.
    """

    closest_t = (np.abs(t_new - np.array(times))).argmin()
    if t_new < times[1]:
        closest_left_t = 0
    elif closest_t == len(times) - 1:
        closest_left_t = len(times) - 2
    else:
        closest_left_t = closest_t
    return closest_left_t
