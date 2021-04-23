import numpy as np

from probnum import _randomvariablelist, diffeq, randvars


class PerturbedStepSolution(diffeq.ODESolution):
    """Output of NoisyStepSolver."""

    def __init__(self, scales: np.array, times: np.array, rvs, interpolants):
        self.scales = scales
        self.times = times
        self.ys = rvs
        self.interpolants = interpolants

    def __call__(self, t):
        if not np.isscalar(t):

            # recursive evaluation (t can now be any array, not just length 1!)
            return _randomvariablelist._RandomVariableList(
                [self.__call__(t_pt) for t_pt in np.asarray(t)]
            )

        # Access last state directly.
        if t == self.times[-1]:
            res = self.ys[-1]
        else:

            # get index of closest left interpolant (í.e. correct interpolant).
            closest_left_t = get_interpolant(self.times, t)
            interpolant = self.interpolants[closest_left_t]
            if self.times[closest_left_t] == t:
                res = randvars.Constant(interpolant(t))
            else:
                relative_time = (t - self.times[closest_left_t]) * self.scales[
                    closest_left_t
                ]
                res = randvars.Constant(
                    interpolant(self.times[closest_left_t] + relative_time)
                )
        return res

    @property
    def locations(self):
        """Time points of the discrete-time solution."""
        return np.array(self.times)

    @property
    def states(self):
        """Discrete-time solution."""
        return _randomvariablelist._RandomVariableList(self.ys)


def get_interpolant(times, t_new):
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
