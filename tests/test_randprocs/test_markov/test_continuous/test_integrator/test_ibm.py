"""Tests for integrated Brownian motion processes."""

import pytest

from probnum import randprocs
from tests.test_randprocs.test_markov.test_continuous import test_sde
from tests.test_randprocs.test_markov.test_continuous.test_integrator import (
    test_integrator,
)


class TestIBM(test_sde.TestLTISDE, test_integrator.TestIntegrator):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        some_ordint,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        self.some_ordint = some_ordint
        spatialdim = 1  # make tests compatible with some_normal_rv1, etc.
        self.transition = randprocs.markov.continuous.integrator.IBM(
            ordint=self.some_ordint,
            spatialdim=spatialdim,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.G = lambda t: self.transition.driftmat
        self.v = lambda t: self.transition.forcevec
        self.L = lambda t: self.transition.dispmat

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    @property
    def integrator(self):
        return self.transition


@pytest.fixture
def dt():
    return 0.1


@pytest.fixture
def ah_22_ibm(dt):
    return np.array(
        [
            [1.0, dt, dt ** 2 / 2.0],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def qh_22_ibm(dt):
    return np.array(
        [
            [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
            [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
        ]
    )


@pytest.fixture
def spdmat3x3(rng):
    return random_spd_matrix(rng, dim=3)


@pytest.fixture
def normal_rv3x3(spdmat3x3):

    return randvars.Normal(
        mean=np.random.rand(3),
        cov=spdmat3x3,
        cov_cholesky=np.linalg.cholesky(spdmat3x3),
    )


class TestIBMValues:

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        spatialdim = 1  # make tests compatible with some_normal_rv1, etc.
        self.transition = randprocs.markov.continuous.integrator.IBM(
            ordint=2,
            spatialdim=spatialdim,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

    def test_discretise_values(self, ah_22_ibm, qh_22_ibm, dt):
        discrete_model = self.transition.discretise(dt=dt)
        np.testing.assert_allclose(discrete_model.state_trans_mat, ah_22_ibm)
        np.testing.assert_allclose(discrete_model.proc_noise_cov_mat, qh_22_ibm)

    def test_forward_rv_values(self, normal_rv3x3, diffusion, ah_22_ibm, qh_22_ibm, dt):
        rv, _ = self.transition.forward_rv(
            normal_rv3x3, t=0.0, dt=dt, _diffusion=diffusion
        )
        np.testing.assert_allclose(ah_22_ibm @ normal_rv3x3.mean, rv[:3].mean)
        np.testing.assert_allclose(
            ah_22_ibm @ normal_rv3x3.cov @ ah_22_ibm.T + diffusion * qh_22_ibm,
            rv.cov,
        )

    def test_forward_realization_values(
        self, normal_rv3x3, diffusion, ah_22_ibm, qh_22_ibm, dt
    ):
        real = normal_rv3x3.mean
        rv, _ = self.transition.forward_realization(
            real, t=0.0, dt=dt, _diffusion=diffusion
        )
        np.testing.assert_allclose(ah_22_ibm @ real, rv.mean)
        np.testing.assert_allclose(diffusion * qh_22_ibm, rv.cov)
