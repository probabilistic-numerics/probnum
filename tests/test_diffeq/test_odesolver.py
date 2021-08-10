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


class TestTimeStopper:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_stops = [2.0, 3.0, 4.0, 5.0]
        self.discrete_events = diffeq._odesolver._TimeStopper(locations=self.time_stops)

        def dummy_perform_step(state, dt, steprule):
            return state, dt

        self.dummy_perform_step = dummy_perform_step

    def test_adjust_dt_to_time_stops(self):
        # Should interfere dt to 0.1 instead of 5.0, because 2 is in self.time_stops
        dt = self.discrete_events.adjust_dt_to_time_stops(t=1.9, dt=5.0)
        assert dt == pytest.approx(0.1)

        # Should not interfere dt if there is no proximity to an event
        dt = self.discrete_events.adjust_dt_to_time_stops(t=1.0, dt=0.00001)
        assert dt == pytest.approx(0.00001)


if __name__ == "__main__":
    unittest.main()
