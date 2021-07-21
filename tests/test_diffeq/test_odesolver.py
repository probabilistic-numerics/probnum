import unittest

import numpy as np

from probnum import diffeq
from probnum.problems.zoo.diffeq import logistic
from probnum.randvars import Constant


class MockODESolver(diffeq.ODESolver):
    """Euler method as an ODE solver."""

    def initialize(self):
        return self.State(
            rv=Constant(self.ivp.y0),
            t=self.ivp.t0,
            error_estimate=np.nan,
            reference_state=None,
        )

    def step(self, state, dt):
        t, x = state.t, state.rv.mean
        xnew = x + dt * self.ivp.f(t, x)

        # return nan as error estimate to ensure that it is not used
        new_state = self.State(
            rv=Constant(xnew), t=t + dt, error_estimate=np.nan, reference_state=xnew
        )
        return new_state

    def rvlist_to_odesol(self, times, rvs):
        return diffeq.ODESolution(locations=times, states=rvs)


class ODESolverTestCase(unittest.TestCase):
    """An ODE Solver has to work with just step() and initialise() provided.

    We implement Euler in MockODESolver to assure this.
    """

    def setUp(self):
        y0 = np.array([0.3])
        ivp = logistic(t0=0, tmax=4, y0=y0)
        euler_order = 1
        self.solver = MockODESolver(ivp, order=euler_order)
        self.step = 0.2

    def test_solve(self):
        steprule = diffeq.stepsize.ConstantSteps(self.step)
        odesol = self.solver.solve(
            steprule=steprule,
        )  # this is the actual part of the test

        # quick check that the result is sensible
        self.assertAlmostEqual(odesol.locations[-1], self.solver.ivp.tmax)
        self.assertAlmostEqual(odesol.states[-1].mean[0], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
