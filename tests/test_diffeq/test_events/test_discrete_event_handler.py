"""Tests for discrete event handlers."""


import pytest

from probnum import diffeq
from tests.test_diffeq.test_events import _event_handler_test_interface


class TestDiscreteEventHandler(_event_handler_test_interface.EventHandlerTest):
    """Tests for discrete event handlers."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_stamps = [2.0, 3.0, 4.0, 5.0]

        def modify(state):
            return diffeq.ODESolver.State(rv=2 * state.rv, t=state.t)

        def condition(state):
            return state.t < 0

        self.discrete_events = diffeq.events.DiscreteEventHandler(
            modify=modify, condition=condition
        )

        def dummy_perform_step(state, dt, steprule):
            return state, dt

        self.dummy_perform_step = dummy_perform_step

    def test_call(self):
        """Test whether __call__ wraps the function correctly."""

        # t > 0, hence the state is not affected
        dummy_state = diffeq.ODESolver.State(rv=3.0, t=1.0)
        updated = self.discrete_events(state=dummy_state)
        assert updated.rv == 3.0

        # t < 0, hence the state is multiplied by two
        dummy_state = diffeq.ODESolver.State(rv=3.0, t=-1.0)
        updated = self.discrete_events(state=dummy_state)
        assert updated.rv == 6.0
