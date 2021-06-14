"""Tests for extended Kalman filtering."""

import pytest

from probnum import filtsmooth

from .. import filtsmooth_testcases as cases
from ._linearization_test_interface import InterfaceTestDiscreteLinearization


class TestDiscreteEKFComponent(InterfaceTestDiscreteLinearization):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.linearizing_component = filtsmooth.DiscreteEKFComponent
        self.linearizing_function_regression_problem = (
            filtsmooth.linearize_regression_problem_ekf
        )


class TestContinuousEKFComponent(cases.LinearisedContinuousTransitionTestCase):
    """Implementation incomplete, hence check that an error is raised."""

    def setUp(self):
        self.linearising_component_benes_daum = filtsmooth.ContinuousEKFComponent
        self.visualise = False
