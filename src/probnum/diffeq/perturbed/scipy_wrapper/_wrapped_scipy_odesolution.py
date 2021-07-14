"""Make a ProbNum ODE solution out of a scipy ODE solution."""
import numpy as np
from scipy.integrate._ivp.common import OdeSolution

from probnum import _randomvariablelist, randvars
from probnum.diffeq import _odesolution
from probnum.filtsmooth._timeseriesposterior import DenseOutputValueType
from probnum.typing import DenseOutputLocationArgType


class WrappedScipyODESolution(_odesolution.ODESolution):
    """ODE solution corresponding to the :class:`WrappedScipyRungeKutta`."""

    def __init__(self, scipy_solution: OdeSolution, rvs: list):
        self.scipy_solution = scipy_solution

        # rvs is of the type `list` of `RandomVariable` and can therefore be
        # directly transformed into a _RandomVariableList
        rv_states = _randomvariablelist._RandomVariableList(rvs)
        super().__init__(locations=scipy_solution.ts, states=rv_states)

    def __call__(self, t: DenseOutputLocationArgType) -> DenseOutputValueType:
        """Evaluate the time-continuous solution at time t.

        Parameters
        ----------
        t
        Location / time at which to evaluate the continuous ODE solution.

        Returns
        -------
        randvars.RandomVariable or _randomvariablelist._RandomVariableList
            Estimate of the states at time ``t`` based on a fourth order polynomial.
        """

        states = self.scipy_solution(t).T
        if np.isscalar(t):
            solution_as_rv = randvars.Constant(states)
        else:
            solution_as_rv = _randomvariablelist._RandomVariableList(
                [randvars.Constant(state) for state in states]
            )
        return solution_as_rv
