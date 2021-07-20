"""Test for ODE residual information operator."""

import numpy as np
import pytest

from probnum import diffeq, filtsmooth, randvars, statespace
from tests.test_diffeq.test_odefiltsmooth.test_information_operators import (
    _information_operator_test_inferface,
)


class TestODEResidual(_information_operator_test_inferface.ODEInformationOperatorTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        prior_ordint = 3
        prior_spatialdim = 2

        self.info_op = diffeq.odefiltsmooth.information_operators.ODEResidual(
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
        # Nothin happens unless an ODE has been incorporated
        with pytest.raises(ValueError):
            self.info_op.as_transition()

        # Basic functionality works
        self.info_op.incorporate_ode(ode=fitzhughnagumo)
        transition = self.info_op.as_transition()
        assert isinstance(transition, statespace.DiscreteGaussian)

        # meascov-fun and meascov-cholesky-fun accepted
        meascov_fun = lambda t: np.eye(self.info_op.output_dim)
        meascov_cholesky_fun = lambda t: np.eye(self.info_op.output_dim)
        transition = self.info_op.as_transition(
            measurement_cov_fun=meascov_fun,
            measurement_cov_cholesky_fun=meascov_cholesky_fun,
        )
        assert isinstance(transition, statespace.DiscreteGaussian)
        assert np.linalg.norm(transition.proc_noise_cov_cholesky_fun(0.0)) > 0.0
        assert np.linalg.norm(transition.proc_noise_cov_mat_fun(0.0)) > 0.0

        # Only meascov-fun accepted
        transition = self.info_op.as_transition(
            measurement_cov_fun=meascov_fun, measurement_cov_cholesky_fun=None
        )
        assert isinstance(transition, statespace.DiscreteGaussian)
        assert np.linalg.norm(transition.proc_noise_cov_mat_fun(0.0)) > 0.0

        # Only meascov-cholesky-fun rejected
        with pytest.raises(ValueError):
            self.info_op.as_transition(
                measurement_cov_fun=None,
                measurement_cov_cholesky_fun=meascov_cholesky_fun,
            )

    @pytest.mark.parametrize("forw_impl", ["sqrt", "classic"])
    @pytest.mark.parametrize("backw_impl", ["sqrt", "classic", "joseph"])
    def test_as_ekf_component(self, fitzhughnagumo, forw_impl, backw_impl):
        # Nothin happens unless an ODE has been incorporated
        with pytest.raises(ValueError):
            self.info_op.as_ekf_component()

        # Basic functionality works
        self.info_op.incorporate_ode(ode=fitzhughnagumo)
        ekf_component = self.info_op.as_ekf_component()
        assert isinstance(
            ekf_component, filtsmooth.gaussian.approx.DiscreteEKFComponent
        )

        # meascov-fun and meascov-cholesky-fun accepted
        meascov_fun = lambda t: np.eye(self.info_op.output_dim)
        meascov_cholesky_fun = lambda t: np.eye(self.info_op.output_dim)
        ekf_component = self.info_op.as_ekf_component(
            measurement_cov_fun=meascov_fun,
            measurement_cov_cholesky_fun=meascov_cholesky_fun,
        )
        assert isinstance(
            ekf_component, filtsmooth.gaussian.approx.DiscreteEKFComponent
        )
        assert np.linalg.norm(ekf_component.proc_noise_cov_cholesky_fun(0.0)) > 0.0
        assert np.linalg.norm(ekf_component.proc_noise_cov_mat_fun(0.0)) > 0.0

        # Only meascov-fun accepted
        ekf_component = self.info_op.as_ekf_component(
            measurement_cov_fun=meascov_fun, measurement_cov_cholesky_fun=None
        )
        assert isinstance(
            ekf_component, filtsmooth.gaussian.approx.DiscreteEKFComponent
        )
        assert np.linalg.norm(ekf_component.proc_noise_cov_mat_fun(0.0)) > 0.0

        # Only meascov-cholesky-fun rejected
        with pytest.raises(ValueError):
            self.info_op.as_ekf_component(
                measurement_cov_fun=None,
                measurement_cov_cholesky_fun=meascov_cholesky_fun,
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
