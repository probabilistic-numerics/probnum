"""Tests for integrated Wiener processes."""

import numpy as np
import pytest

from probnum import config, randprocs, randvars
from probnum.problems.zoo import linalg as linalg_zoo
from tests.test_randprocs.test_markov.test_continuous import test_lti_sde
from tests.test_randprocs.test_markov.test_integrator import integrator_test_mixin


@pytest.mark.parametrize("initarg", [0.0, 2.0])
@pytest.mark.parametrize("num_derivatives", [0, 1, 4])
@pytest.mark.parametrize("wiener_process_dimension", [1, 2, 3])
@pytest.mark.parametrize("use_initrv", [True, False])
@pytest.mark.parametrize("diffuse", [True, False])
def test_iwp_construction(
    initarg, num_derivatives, wiener_process_dimension, use_initrv, diffuse
):
    if use_initrv:
        d = (num_derivatives + 1) * wiener_process_dimension
        initrv = randvars.Normal(np.arange(d), np.diag(np.arange(1, d + 1)))
    else:
        initrv = None
    if use_initrv and diffuse:
        with pytest.warns(Warning):
            randprocs.markov.integrator.IntegratedWienerProcess(
                initarg=initarg,
                num_derivatives=num_derivatives,
                wiener_process_dimension=wiener_process_dimension,
                initrv=initrv,
                diffuse=diffuse,
            )

    else:
        iwp = randprocs.markov.integrator.IntegratedWienerProcess(
            initarg=initarg,
            num_derivatives=num_derivatives,
            wiener_process_dimension=wiener_process_dimension,
            initrv=initrv,
            diffuse=diffuse,
        )

        assert isinstance(iwp, randprocs.markov.integrator.IntegratedWienerProcess)
        assert isinstance(iwp, randprocs.markov.MarkovProcess)
        assert isinstance(
            iwp.transition,
            randprocs.markov.integrator.IntegratedWienerTransition,
        )


class TestIntegratedWienerTransition(
    test_lti_sde.TestLTISDE, integrator_test_mixin.IntegratorMixInTestMixIn
):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        some_num_derivatives,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        self.some_num_derivatives = some_num_derivatives
        wiener_process_dimension = 1  # make tests compatible with some_normal_rv1, etc.
        self.transition = randprocs.markov.integrator.IntegratedWienerTransition(
            num_derivatives=self.some_num_derivatives,
            wiener_process_dimension=wiener_process_dimension,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.G = lambda t: self.transition.drift_matrix
        self.v = lambda t: self.transition.force_vector
        self.L = lambda t: self.transition.dispersion_matrix

        self.G_const = self.transition.drift_matrix
        self.v_const = self.transition.force_vector
        self.L_const = self.transition.dispersion_matrix

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)
        self.l = lambda t, x: self.L(t)

    @property
    def integrator(self):
        return self.transition

    def test_wiener_process_dimension(self, test_ndim):
        assert self.transition.wiener_process_dimension == 1

    def test_discretise_no_force(self):
        """LTISDE.discretise() works if there is zero force (there is an "if" in the
        fct)."""

        # Sanity checks: if this does not work, the test is meaningless
        np.testing.assert_allclose(self.transition.force_vector_function(0.0), 0.0)
        np.testing.assert_allclose(self.transition.force_vector, 0.0)

        # Test discretisation
        out = self.transition.discretise(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.LTIGaussian)


class TestIBMLinOps(
    test_lti_sde.TestLTISDE, integrator_test_mixin.IntegratorMixInTestMixIn
):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        some_num_derivatives,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        self.some_num_derivatives = some_num_derivatives
        spatialdim = 1  # make tests compatible with some_normal_rv1, etc.
        with config(lazy_linalg=True):
            self.transition = randprocs.markov.integrator.IntegratedWienerTransition(
                num_derivatives=self.some_num_derivatives,
                wiener_process_dimension=spatialdim,
                forward_implementation=forw_impl_string_linear_gauss,
                backward_implementation=backw_impl_string_linear_gauss,
            )

        self.G = lambda t: self.transition.drift_matrix
        self.v = lambda t: self.transition.force_vector
        self.L = lambda t: self.transition.dispersion_matrix

        self.G_const = self.transition.drift_matrix
        self.v_const = self.transition.force_vector
        self.L_const = self.transition.dispersion_matrix

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)
        self.l = lambda t, x: self.L(t)

    @property
    def integrator(self):
        return self.transition

    def test_wiener_process_dimension(self, test_ndim):
        assert self.transition.wiener_process_dimension == 1

    def test_drift(self, some_normal_rv1):
        expected = self.g(0.0, some_normal_rv1.mean)
        received = self.transition.drift_function(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_dispersionmatrix(self, some_normal_rv1):
        expected = self.l(0.0, some_normal_rv1.mean)
        received = self.transition.dispersion_function(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received.todense(), expected.todense())

    def test_drift_jacobian(self, some_normal_rv1):
        expected = self.dg(0.0, some_normal_rv1.mean)
        received = self.transition.drift_jacobian(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received.todense(), expected.todense())

    def test_drift_matrix_function(self):
        expected = self.G(0.0)
        received = self.transition.drift_matrix_function(0.0)
        np.testing.assert_allclose(expected.todense(), received.todense())

    def test_dispersion_matrix_function(self):
        expected = self.L(0.0)
        received = self.transition.dispersion_matrix_function(0.0)
        np.testing.assert_allclose(expected.todense(), received.todense())

    def test_drift_matrix(self):
        np.testing.assert_allclose(
            self.transition.drift_matrix.todense(), self.G_const.todense()
        )

    def test_force_vector(self):
        np.testing.assert_allclose(self.transition.force_vector, self.v_const)

    def test_dispersion_matrix(self):
        np.testing.assert_allclose(
            self.transition.dispersion_matrix.todense(), self.L_const.todense()
        )

    def test_discretise_no_force(self):
        """LTISDE.discretise() works if there is zero force (there is an "if" in the
        fct)."""
        # Sanity checks: if this does not work, the test is meaningless
        np.testing.assert_allclose(self.transition.force_vector_function(0.0), 0.0)
        np.testing.assert_allclose(self.transition.force_vector, 0.0)

        # Test discretisation
        out = self.transition.discretise(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.LTIGaussian)


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
    return linalg_zoo.random_spd_matrix(rng, dim=3)


@pytest.fixture
def normal_rv3x3(spdmat3x3):

    return randvars.Normal(
        mean=np.random.rand(3),
        cov=spdmat3x3,
        cov_cholesky=np.linalg.cholesky(spdmat3x3),
    )


def test_iwp_transition_drift_matrix_values():
    F = randprocs.markov.integrator.IntegratedWienerTransition._iwp_drift_matrix(
        num_derivatives=2,
        wiener_process_dimension=1,
    )
    expected = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(F, expected)


class TestIntegratedWienerTransitionValues:

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):
        wiener_process_dimension = 1  # make tests compatible with some_normal_rv1, etc.
        self.transition = randprocs.markov.integrator.IntegratedWienerTransition(
            num_derivatives=2,
            wiener_process_dimension=wiener_process_dimension,
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
