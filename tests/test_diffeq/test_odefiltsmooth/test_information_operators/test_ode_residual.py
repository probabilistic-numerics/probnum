"""Test for ODE residual information operator."""

import numpy as np
import pytest

from probnum import diffeq, filtsmooth, randvars, statespace
from tests.test_diffeq.test_odefiltsmooth.test_information_operators import (
    _information_operator_test_inferface,
)


class TestExplicitODEResidual(
    _information_operator_test_inferface.ODEInformationOperatorTest
):
    @pytest.fixture(autouse=True)
    def _setup(self):
        prior_ordint = 3
        prior_spatialdim = 2

        self.info_op = diffeq.odefiltsmooth.information_operators.ExplicitODEResidual(
            prior_ordint=prior_ordint, prior_spatialdim=prior_spatialdim
        )
        self.initial_rv = randvars.Normal(
            mean=np.arange(self.info_op.input_dim), cov=np.eye(self.info_op.input_dim)
        )

    def test_call(self, fitzhughnagumo):
        self.info_op.incorporate_ode(ode=fitzhughnagumo)
        called = self.info_op(fitzhughnagumo.t0, self.initial_rv.mean)
        assert isinstance(called, np.ndarray)
        assert called.shape == (self.info_op.output_dim,)

    def test_jacobian(self, fitzhughnagumo):
        self.info_op.incorporate_ode(ode=fitzhughnagumo)
        called = self.info_op.jacobian(fitzhughnagumo.t0, self.initial_rv.mean)
        assert isinstance(called, np.ndarray)
        assert called.shape == (self.info_op.output_dim, self.info_op.input_dim)

    def test_as_transition(self, fitzhughnagumo):
        self.info_op.incorporate_ode(ode=fitzhughnagumo)
        ekf_component = self.info_op.as_ekf_component()
        assert isinstance(ekf_component, statespace.DiscreteGaussian)

    def test_as_ekf_component(self, fitzhughnagumo):
        self.info_op.incorporate_ode(ode=fitzhughnagumo)
        ekf_component = self.info_op.as_ekf_component()
        assert isinstance(
            ekf_component, filtsmooth.gaussian.approx.DiscreteEKFComponent
        )

    def test_incorporate_ode(self, fitzhughnagumo):
        self.info_op.incorporate_ode(ode=fitzhughnagumo)
        assert self.info_op.ode == fitzhughnagumo

        # Incorporating an ODE when another one has been
        # incorporated raises a ValueError
        with pytest.raises(ValueError):
            self.info_op.incorporate_ode(ode=fitzhughnagumo)

    def test_ode_has_been_incorporated(self, fitzhughnagumo):
        assert self.info_op.ode_has_been_incorporated is False
        self.info_op.incorporate_ode(ode=fitzhughnagumo)
        assert self.info_op.ode_has_been_incorporated is True
