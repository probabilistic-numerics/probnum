import numpy as np

from probnum import _randomvariablelist, diffeq, randvars
from probnum.diffeq.odefiltsmooth import kalman_odesolution


class NoisyStepSolution(diffeq.ODESolution):
    """Output of NoisyStepSolver.

    Parameters
    ----------
    times
        array of length N+1 where N is the number of steps that the ODE solver has taken.
    interpolants
        array of scipy.DenseOutput objects, shape (N,). Interpolates
        the deterministic
        solver output.
    kalman_odesolutions
        array of probnum.diffeq.KalmanODESolution objects, shape (N,).
        Interpolates the perturbations.
    """

    def __init__(self, projected_times, evaluated_times, states, interpolants):
        self.projected_times = projected_times
        self.evaluated_times = evaluated_times
        self.states = states
        self.interpolants = interpolants

    def __call__(self, t):
        if not np.isscalar(t):
            # recursive evaluation (t can now be any array, not just length 1!)
            return _randomvariablelist._RandomVariableList(
                [self.__call__(t_pt) for t_pt in np.asarray(t)]
            )
        # find closest left timepoint (í.e. correct interpolant) of evaluation.
        closest_left_t = self.find_closest_left_element(self.projected_times, t)
        # position within in the perturbed interval
        interval_pos = t - self.projected_times[closest_left_t]
        # size of the perturbed interval
        if t != self.projected_times[-1]:
            interval_size = (
                self.projected_times[closest_left_t + 1]
                - self.projected_times[closest_left_t]
            )
            # relative position in the interval of the projected evaluation (i.e. the position that we care about within the intervall)
            relative_pos = interval_pos / interval_size
            # position in the interval of the perturbed projection
            new_pos = (
                relative_pos
                * (
                    self.evaluated_times[closest_left_t + 1]
                    - self.evaluated_times[closest_left_t]
                )
                + self.projected_times[closest_left_t]
            )
            interpolant = self.interpolants[closest_left_t]
            # evalution at timepoint t, not the interpolants' timepoint
            interpolation = randvars.Constant(interpolant(new_pos))
        # for the last element (as there's no interpolant that can be evaluated)
        else:
            interpolation = self.states[-1]
        return interpolation

    @property
    def t(self):
        """Time points of the discrete-time solution."""
        return self.projected_times

    @property
    def y(self):
        """Discrete-time solution."""
        return _randomvariablelist._RandomVariableList(self.states)

    def __len__(self) -> int:
        """Number of points in the discrete-time solution."""
        return len(self.states)

    def __getitem__(self, idx: int) -> _randomvariablelist.RandomVariable:
        """Access the :math:`i`th element of the discrete-time solution."""
        return self.states[idx]

    def sample(self, t=None, size=()):
        return "Sampling not possible"

    def find_closest_left_element(self, times, t_new):
        """

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
        # find closest timepoint of evaluation
        closest_t = (np.abs(t_new - np.array(times))).argmin()
        # if t_new is in the first interpolant
        if t_new < times[1]:
            closest_left_t = 0
        # make sure that the point is on the left of the evaluation point
        elif t_new < times[closest_t]:
            closest_left_t = closest_t - 1
        else:
            closest_left_t = closest_t
        return closest_left_t
