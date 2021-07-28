import unittest

import numpy as np

from probnum import diffeq
from probnum.problems.zoo.diffeq import logistic
from probnum.randvars import Constant


class MockODESolver(diffeq.ODESolver):
    """Euler method as an ODE solver."""

    def __init__(self, *args, **kwargs):
        self.ivp = None
        super().__init__(*args, **kwargs)

    def initialise(self, ivp):
        self.ivp = ivp
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
        self.assertAlmostEqual(odesol.locations[-1], self.solver.ivp.tmax)
        self.assertAlmostEqual(odesol.states[-1].mean[0], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
