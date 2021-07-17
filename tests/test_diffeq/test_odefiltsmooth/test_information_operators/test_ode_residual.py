"""Test for ODE residual information operator."""

import pytest

from tests.test_diffeq.test_odefiltsmooth.test_information_operators import (
    _information_operator_test_inferface,
)


class TestODEResidual(_information_operator_test_inferface.ODEInformationOperatorTest):
    def test_call(self):
        raise NotImplementedError

    def test_jacobian(self):
        raise NotImplementedError

    def test_as_transition(self):
        raise NotImplementedError

    def test_as_ekf_component(self):
        raise NotImplementedError

    def test_incorporate_ode(self):
        raise NotImplementedError

    def test_ode_has_been_incorporated(self):
        raise NotImplementedError
