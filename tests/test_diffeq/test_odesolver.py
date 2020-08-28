import unittest

import numpy as np

from probnum.diffeq import logistic, ODESolver, ConstantSteps
from probnum.random_variables import Dirac


class MockODESolver(ODESolver):
    """Euler method as an ODE solver"""

    def initialise(self):
        return self.ivp.t0, self.ivp.initrv

    def step(self, start, stop, current):
        h = stop - start
        x = current.mean
        xnew = x + h * self.ivp(start, x)
        return (
            Dirac(xnew),
            np.nan,
        )  # return nan as error estimate to ensure that it is not used


class ODESolverTestCase(unittest.TestCase):
    """
    An ODE Solver has to work with just step() and initialise() provided.
    We implement Euler in MockODESolver to assure this.
    """

    def setUp(self):
        y0 = Dirac(0.3)
        ivp = logistic([0, 4], initrv=y0)
        self.solver = MockODESolver(ivp)
        self.step = 0.2

    def test_solve(self):
        steprule = ConstantSteps(self.step)
        odesol = self.solver.solve(
            firststep=self.step, steprule=steprule
        )  # this is the actual part of the test

        # quick check that the result is sensible
        self.assertAlmostEqual(odesol.t[-1], self.solver.ivp.tmax)
        self.assertAlmostEqual(odesol.y[-1].mean, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
