"""Test for ODE residual information operator."""

import numpy as np
import pytest

from probnum import diffeq, randprocs, randvars
from tests.test_diffeq.test_odefiltsmooth.test_information_operators import (
    _information_operator_test_inferface,
)


class TestODEResidual(_information_operator_test_inferface.ODEInformationOperatorTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        num_prior_derivatives = 3
        ode_dimension = 2

        self.info_op = diffeq.odefiltsmooth.information_operators.ODEResidual(
            num_prior_derivatives=num_prior_derivatives, ode_dimension=ode_dimension
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
        assert isinstance(transition, randprocs.markov.discrete.DiscreteGaussian)

        # meascov-fun and meascov-cholesky-fun accepted
        meascov_fun = lambda t: np.eye(self.info_op.output_dim)
        meascov_cholesky_fun = lambda t: np.eye(self.info_op.output_dim)
        transition = self.info_op.as_transition(
            measurement_cov_fun=meascov_fun,
            measurement_cov_cholesky_fun=meascov_cholesky_fun,
        )
        assert isinstance(transition, randprocs.markov.discrete.DiscreteGaussian)
        assert np.linalg.norm(transition.proc_noise_cov_cholesky_fun(0.0)) > 0.0
        assert np.linalg.norm(transition.proc_noise_cov_mat_fun(0.0)) > 0.0

        # Only meascov-fun accepted
        transition = self.info_op.as_transition(
            measurement_cov_fun=meascov_fun, measurement_cov_cholesky_fun=None
        )
        assert isinstance(transition, randprocs.markov.discrete.DiscreteGaussian)
        assert np.linalg.norm(transition.proc_noise_cov_mat_fun(0.0)) > 0.0

        # Only meascov-cholesky-fun rejected
        with pytest.raises(ValueError):
            self.info_op.as_transition(
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
