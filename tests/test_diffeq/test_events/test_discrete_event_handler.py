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
        wrapped = self.discrete_events(self.dummy_perform_step)

        # t > 0, hence the state is not affected
        dummy_state = diffeq.ODESolver.State(rv=3.0, t=1.0)
        non_wrapped_output = self.dummy_perform_step(
            state=dummy_state, dt=0.1, steprule=None
        )
        wrapped_output = wrapped(state=dummy_state, dt=0.1, steprule=None)
        assert non_wrapped_output[0].rv == 3.0
        assert wrapped_output[0].rv == 3.0

        # t < 0, hence the state is multiplied by two
        dummy_state = diffeq.ODESolver.State(rv=3.0, t=-1.0)
        non_wrapped_output = self.dummy_perform_step(
            state=dummy_state, dt=0.1, steprule=None
        )
        wrapped_output = wrapped(state=dummy_state, dt=0.1, steprule=None)
        assert non_wrapped_output[0].rv == 3.0
        assert wrapped_output[0].rv == 6.0
