"""Tests for unscented Kalman filtering."""

import pytest

from probnum import filtsmooth, statespace

from ._linearization_test_interface import InterfaceDiscreteLinearizationTest


class TestContinuousUKFComponent:
    """Implementation incomplete, hence check that an error is raised."""

    def test_notimplementederror(self):
        sde = statespace.SDE(1, None, None, None)  # content is irrelevant.
        with pytest.raises(NotImplementedError):
            filtsmooth.ContinuousUKFComponent(sde)


class TestDiscreteUKFComponent(InterfaceDiscreteLinearizationTest):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.linearizing_component = filtsmooth.DiscreteUKFComponent
