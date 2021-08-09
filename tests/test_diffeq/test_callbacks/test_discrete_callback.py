"""Tests for discrete event handlers."""


import dataclasses

import pytest

from probnum import diffeq
from tests.test_diffeq.test_callbacks import _callback_test_interface


class TestDiscreteCallback(_callback_test_interface.CallbackTest):
    """Tests for discrete event handlers."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_stops = [2.0, 3.0, 4.0, 5.0]

        def replace(state):
            return dataclasses.replace(state, rv=2 * state.rv)

        def condition(state):
            return state.t < 0

        self.discrete_callbacks = diffeq.callbacks.DiscreteCallback(
            replace=replace, condition=condition
        )

        def dummy_perform_step(state, dt, steprule):
            return state, dt

        self.dummy_perform_step = dummy_perform_step

    def test_call(self):
        """Test whether __call__ wraps the function correctly."""

        # t > 0, hence the state is not affected
        dummy_state = diffeq.ODESolverState(ivp=None, rv=3.0, t=1.0)
        updated = self.discrete_callbacks(state=dummy_state)
        assert updated.rv == 3.0

        # t < 0, hence the state is multiplied by two
        dummy_state = diffeq.ODESolverState(ivp=None, rv=3.0, t=-1.0)
        updated = self.discrete_callbacks(state=dummy_state)
        assert updated.rv == 6.0
