"""Tests for EK0/1."""

import numpy as np
import pytest

from probnum import diffeq, filtsmooth
from tests.test_diffeq.test_odefiltsmooth.test_approx_strategies import (
    _approx_test_interface,
)


class TestEK0(_approx_test_interface.ApproximationStrategyTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.approx_strategy = diffeq.odefiltsmooth.approx_strategies.EK0()
        self.info_op = diffeq.odefiltsmooth.information_operators.ODEResidual(
            num_prior_derivatives=3, ode_dimension=2
        )

    def test_call(self, fitzhughnagumo):

        # not-yet-registered ODE raises errors
        with pytest.raises(ValueError):
            self.approx_strategy(self.info_op)

        self.info_op.incorporate_ode(ode=fitzhughnagumo)

        called = self.approx_strategy(self.info_op)
        assert isinstance(
            called,
            diffeq.odefiltsmooth.information_operators.ApproximateInformationOperator,
        )
        assert isinstance(
            called.as_transition(), filtsmooth.gaussian.approx.DiscreteEKFComponent
        )


class TestEK1(_approx_test_interface.ApproximationStrategyTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.approx_strategy = diffeq.odefiltsmooth.approx_strategies.EK1()
        self.info_op = diffeq.odefiltsmooth.information_operators.ODEResidual(
            num_prior_derivatives=3, ode_dimension=2
        )

    def test_call(self, fitzhughnagumo):

        # not-yet-registered ODE raises errors as soon as the approximate info operator is called.
        with pytest.raises(ValueError):
            approx_transition = self.approx_strategy(self.info_op).as_transition()
            approx_transition.forward_realization(
                np.arange(self.info_op.input_dim), t=0.0
            )

        self.info_op.incorporate_ode(ode=fitzhughnagumo)

        called = self.approx_strategy(self.info_op)
        assert isinstance(
            called,
            diffeq.odefiltsmooth.information_operators.ApproximateInformationOperator,
        )
        assert isinstance(
            called.as_transition(), filtsmooth.gaussian.approx.DiscreteEKFComponent
        )
