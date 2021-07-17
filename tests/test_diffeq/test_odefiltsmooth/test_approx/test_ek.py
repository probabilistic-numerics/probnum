"""Tests for EK0/1."""

import pytest

from probnum import diffeq
from tests.test_diffeq.test_odefiltsmooth.test_approx import _approx_test_interface


class TestEK0(_approx_test_interface.ApproximationStrategyTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.approx_strategy = diffeq.odefiltsmooth.approx.EK0()

    def test_call(self):
        pass
