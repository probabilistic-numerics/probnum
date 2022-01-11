import unittest

import numpy as np
import pytest

from probnum import diffeq
from probnum.problems.zoo.diffeq import logistic
from probnum.randvars import Constant


class MockODESolver(diffeq.ODESolver):
    """Euler method as an ODE solver."""

    def initialize(self, ivp):
        return diffeq.ODESolverState(
            ivp=ivp,
            rv=Constant(ivp.y0),
            t=ivp.t0,
            error_estimate=np.nan,
            reference_state=None,
        )

    def attempt_step(self, state, dt):
        t, x = state.t, state.rv.mean
        xnew = x + dt * state.ivp.f(t, x)

        # return nan as error estimate to ensure that it is not used
        new_state = diffeq.ODESolverState(
            ivp=state.ivp,
            rv=Constant(xnew),
            t=t + dt,
            error_estimate=np.nan,
            reference_state=xnew,
        )
        return new_state

    def rvlist_to_odesol(self, times, rvs):
        return diffeq.ODESolution(locations=times, states=rvs)


class ODESolverTestCase(unittest.TestCase):
    """An ODE Solver has to work with just step() and initialise() provided.

    We implement Euler in MockODESolver to assure this.
    """

    def setUp(self):
        step = 0.2
        steprule = diffeq.stepsize.ConstantSteps(step)
        euler_order = 1
        self.solver = MockODESolver(steprule=steprule, order=euler_order)

    def test_solve(self):
        y0 = np.array([0.3])
        ivp = logistic(t0=0, tmax=4, y0=y0)
        odesol = self.solver.solve(
            ivp=ivp,
        )  # this is the actual part of the test

        # quick check that the result is sensible
        self.assertAlmostEqual(odesol.locations[-1], ivp.tmax)
        self.assertAlmostEqual(odesol.states[-1].mean[0], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
