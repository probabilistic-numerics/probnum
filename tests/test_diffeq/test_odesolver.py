import unittest

import numpy as np

from probnum.diffeq import ConstantSteps, ODESolution, ODESolver
from probnum.problems.zoo.diffeq import logistic
from probnum.randvars import Constant


class MockODESolver(ODESolver):
    """Euler method as an ODE solver."""

    def initialise(self):
        return self.ivp.t0, Constant(self.ivp.y0)

    def step(self, start, stop, current):
        h = stop - start
        x = current.mean
        xnew = x + h * self.ivp.f(start, x)
        return (
            Constant(xnew),
            np.nan,
            xnew,
        )  # return nan as error estimate to ensure that it is not used

    def rvlist_to_odesol(self, times, rvs):
        return ODESolution(locations=times, states=rvs)


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
        steprule = ConstantSteps(self.step)
        odesol = self.solver.solve(
            steprule=steprule,
        )  # this is the actual part of the test

        # quick check that the result is sensible
        self.assertAlmostEqual(odesol.locations[-1], self.solver.ivp.tmax)
        self.assertAlmostEqual(odesol.states[-1].mean[0], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
