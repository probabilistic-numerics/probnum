"""Tests for EK0/1."""

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
            prior_ordint=3, prior_spatialdim=2
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
            prior_ordint=3, prior_spatialdim=2
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