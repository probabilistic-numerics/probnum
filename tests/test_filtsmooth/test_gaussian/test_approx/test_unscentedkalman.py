"""Tests for unscented Kalman filtering."""

from probnum import filtsmooth

from ._linearization_test_interface import InterfaceDiscreteLinearizationTest

import pytest


class TestDiscreteUKFComponent(InterfaceDiscreteLinearizationTest):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.linearizing_component = filtsmooth.gaussian.approx.DiscreteUKFComponent
