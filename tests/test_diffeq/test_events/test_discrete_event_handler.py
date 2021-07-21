"""Tests for the discrete event handler."""

import pytest

from probnum import diffeq
from tests.test_diffeq.test_events import _event_handler_test_interface


class TestDiscreteEventHandler(_event_handler_test_interface.EventHandlerTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_stamps = [2.0, 3.0, 4.0, 5.0]
        self.discrete_events = diffeq.events.DiscreteEventHandler(
            time_stamps=self.time_stamps
        )

        def dummy_perform_step(start, stop, current):
            return current

        self.dummy_perform_step = dummy_perform_step

    def test_call(self):
        wrapped = self.discrete_events(self.dummy_perform_step)

        # Signature remains the same
        assert wrapped(start=0.1, stop=0.2, current=3.0) == 3.0
        # more tests to  come

    def test_interfere_dt(self):
        # Should interfere dt to 0.1 instead of 5.0, because 2 is in self.time_stamps
        dt = self.discrete_events.interfere_dt(t=1.9, dt=5.0)
        assert dt == pytest.approx(0.1)

        # Should not interfere dt if there is no proximity to an event
        dt = self.discrete_events.interfere_dt(t=1.0, dt=0.00001)
        assert dt == pytest.approx(0.00001)

    def test_intervene_state(self):
        """With only time-stamps, intervention does not happen."""
        assert self.discrete_events.intervene_state(0.3) == 0.3
