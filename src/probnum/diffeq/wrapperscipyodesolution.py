"""Make a ProbNum ODE solution out of a scipy ODE solution."""
import numpy as np

from probnum import _randomvariablelist, diffeq, randvars


class WrapperScipyODESolution(diffeq.ODESolution):
    def __init__(self, scipy_solution, locations, states):
        """Evaluate the time-continuous solution at time t.
        Parameters
        ----------
        t : float
            Location / time at which to evaluate the continuous ODE solution.
        Returns
        -------
        :obj:`RandomVariable`
            Probabilistic estimate of the continuous-time solution at time ``t``.
        """
        self.scipy_solution = scipy_solution
        self.locations = locations
        self.states = _randomvariablelist._RandomVariableList(states)

    def __call__(self, t):
        """"solution as _RandomVariableList."""
        states = np.array(self.scipy_solution(t)).T
        solution_as_rv = _randomvariablelist._RandomVariableList(
            list(map(lambda x: (randvars.Constant(x)), states))
        )
        return solution_as_rv

    def __len__(self):
        """Number of points in the discrete-time solution."""
        return len(self.scipy_solution.ts)

    def __getitem__(self, idx):
        """Access the :math:`i`th element of the discrete-time solution."""
        return self.scipy_solution.interpolants[idx](self.scipy_solution.ts[idx])

    def sample(self, t=None, size=()):
        return "Sampling not possible"
