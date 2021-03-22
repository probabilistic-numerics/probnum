import unittest

import numpy as np

from probnum.diffeq import ConstantSteps, ODESolution, ODESolver, logistic
from probnum.randvars import Constant


class MockODESolution(ODESolution):
    def __init__(self, t, y):
        self._t = t
        self._y = y

    @property
    def t(self):
        return self._t

    @property
    def y(self):
        # pylint: disable=invalid-overridden-method
        return self._y


class MockODESolver(ODESolver):
    """Euler method as an ODE solver."""

    def initialise(self):
        return self.ivp.t0, self.ivp.initrv

    def step(self, start, stop, current):
        h = stop - start
        x = current.mean
        xnew = x + h * self.ivp(start, x)
        return (
            Constant(xnew),
            np.nan,
        )  # return nan as error estimate to ensure that it is not used

    def rvlist_to_odesol(self, times, rvs):
        return MockODESolution(times, rvs)


class ODESolverTestCase(unittest.TestCase):
    """An ODE Solver has to work with just step() and initialise() provided.

    We implement Euler in MockODESolver to assure this.
    """

    def setUp(self):
        y0 = Constant(0.3)
        ivp = logistic([0, 4], initrv=y0)
        euler_order = 1
        self.solver = MockODESolver(ivp, order=euler_order)
        self.step = 0.2

    def test_solve(self):
        steprule = ConstantSteps(self.step)
        odesol = self.solver.solve(
            steprule=steprule,
        )  # this is the actual part of the test

        # quick check that the result is sensible
        self.assertAlmostEqual(odesol.t[-1], self.solver.ivp.tmax)
        self.assertAlmostEqual(odesol.y[-1].mean, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
