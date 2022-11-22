import numpy as np

from probnum import config, linops, randprocs, randvars

import pytest
from tests.probnum.randprocs.markov.discrete import test_nonlinear_gaussian


@pytest.fixture(params=["classic", "sqrt"])
def forw_impl_string_linear_gauss(request):
    """Forward implementation choices passed via strings."""
    return request.param


@pytest.fixture(params=["classic", "joseph", "sqrt"])
def backw_impl_string_linear_gauss(request):
    """Backward implementation choices passed via strings."""
    return request.param


class TestLinearGaussian(test_nonlinear_gaussian.TestNonlinearGaussian):
    """Test class for linear Gaussians. Inherits tests from `TestNonlinearGaussian` but
    overwrites the forward and backward transitions.

    Also tests that different forward and backward implementations yield the same
    results.
    """

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        state_dim,
        spdmat1,
        spdmat2,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):

        self.transition_matrix_fun = lambda t: spdmat1
        self.noise_fun = lambda t: randvars.Normal(
            mean=np.arange(state_dim), cov=spdmat2
        )

        self.transition = randprocs.markov.discrete.LinearGaussian(
            input_dim=state_dim,
            output_dim=state_dim,
            transition_matrix_fun=self.transition_matrix_fun,
            noise_fun=self.noise_fun,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.transition_fun = lambda t, x: self.transition_matrix_fun(t) @ x
        self.transition_fun_jacobian = lambda t, x: self.transition_matrix_fun(t)

    # Test access to system matrices

    def test_transition_matrix_fun(self):
        received = self.transition.transition_matrix_fun(0.0)
        expected = self.transition_matrix_fun(0.0)
        np.testing.assert_allclose(received, expected)

    def test_noise_fun(self):
        received = self.transition.noise_fun(0.0)
        expected = self.noise_fun(0.0)
        np.testing.assert_allclose(received.mean, expected.mean)
        np.testing.assert_allclose(received.cov, expected.cov)

    # Test forward and backward implementations

    def test_forward_rv(self, some_normal_rv1):
        out, _ = self.transition.forward_rv(some_normal_rv1, 0.0)
        assert isinstance(out, randvars.Normal)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_rv(some_normal_rv1, some_normal_rv2)
        assert isinstance(out, randvars.Normal)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_realization(
            some_normal_rv1.mean, some_normal_rv2
        )
        assert isinstance(out, randvars.Normal)

    def test_all_forward_rv_same(self, some_normal_rv1, diffusion):
        """Assert all implementations give the same output."""
        out_classic, info_classic = self.transition._forward_rv_classic(
            some_normal_rv1, 0.0, compute_gain=True, _diffusion=diffusion
        )
        out_sqrt, info_sqrt = self.transition._forward_rv_sqrt(
            some_normal_rv1, 0.0, compute_gain=True, _diffusion=diffusion
        )

        np.testing.assert_allclose(out_classic.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_classic.cov, out_sqrt.cov)
        np.testing.assert_allclose(info_classic["crosscov"], info_sqrt["crosscov"])
        np.testing.assert_allclose(info_classic["gain"], info_sqrt["gain"])

    def test_all_backward_rv_same_no_cache(
        self, some_normal_rv1, some_normal_rv2, diffusion
    ):
        """Assert all implementations give the same output -- no gain or forwarded RV
        passed."""

        out_classic, _ = self.transition._backward_rv_classic(
            some_normal_rv1, some_normal_rv2, t=0.0, _diffusion=diffusion
        )
        out_sqrt, _ = self.transition._backward_rv_sqrt(
            some_normal_rv1, some_normal_rv2, t=0.0, _diffusion=diffusion
        )
        out_joseph, _ = self.transition._backward_rv_joseph(
            some_normal_rv1, some_normal_rv2, t=0.0, _diffusion=diffusion
        )

        # Classic -- sqrt
        np.testing.assert_allclose(out_classic.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_classic.cov, out_sqrt.cov)

        # Joseph -- sqrt
        np.testing.assert_allclose(out_joseph.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_joseph.cov, out_sqrt.cov)

    def test_all_backward_rv_same_with_cache(
        self, some_normal_rv1, some_normal_rv2, diffusion
    ):
        """Assert all implementations give the same output -- gain and forwarded RV
        passed."""

        rv_forward, info = self.transition.forward_rv(
            some_normal_rv2, 0.0, compute_gain=True, _diffusion=diffusion
        )
        gain = info["gain"]

        out_classic, _ = self.transition._backward_rv_classic(
            some_normal_rv1,
            some_normal_rv2,
            rv_forwarded=rv_forward,
            gain=gain,
            t=0.0,
            _diffusion=diffusion,
        )
        out_sqrt, _ = self.transition._backward_rv_sqrt(
            some_normal_rv1,
            some_normal_rv2,
            rv_forwarded=rv_forward,
            gain=gain,
            t=0.0,
            _diffusion=diffusion,
        )
        out_joseph, _ = self.transition._backward_rv_joseph(
            some_normal_rv1,
            some_normal_rv2,
            rv_forwarded=rv_forward,
            gain=gain,
            t=0.0,
            _diffusion=diffusion,
        )

        # Classic -- sqrt
        np.testing.assert_allclose(out_classic.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_classic.cov, out_sqrt.cov)

        # Joseph -- sqrt
        np.testing.assert_allclose(out_joseph.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_joseph.cov, out_sqrt.cov)

    def test_all_backward_realization_same_no_cache(
        self, some_normal_rv1, some_normal_rv2, diffusion
    ):
        """Assert all implementations give the same output -- no gain or forwarded RV
        passed."""

        out_classic, _ = self.transition._backward_rv_classic(
            randvars.Constant(some_normal_rv1.mean),
            some_normal_rv2,
            t=0.0,
            _diffusion=diffusion,
        )
        out_sqrt, _ = self.transition._backward_rv_sqrt(
            randvars.Constant(some_normal_rv1.mean),
            some_normal_rv2,
            t=0.0,
            _diffusion=diffusion,
        )
        out_joseph, _ = self.transition._backward_rv_joseph(
            randvars.Constant(some_normal_rv1.mean),
            some_normal_rv2,
            t=0.0,
            _diffusion=diffusion,
        )

        # Classic -- sqrt
        np.testing.assert_allclose(out_classic.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_classic.cov, out_sqrt.cov)

        # Joseph -- sqrt
        np.testing.assert_allclose(out_joseph.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_joseph.cov, out_sqrt.cov)

    def test_all_backward_realization_same_with_cache(
        self, some_normal_rv1, some_normal_rv2, diffusion
    ):
        """Assert all implementations give the same output -- gain and forwarded RV
        passed."""

        rv_forward, info = self.transition.forward_rv(
            some_normal_rv2, 0.0, compute_gain=True, _diffusion=diffusion
        )
        gain = info["gain"]

        out_classic, _ = self.transition._backward_rv_classic(
            randvars.Constant(some_normal_rv1.mean),
            some_normal_rv2,
            rv_forwarded=rv_forward,
            gain=gain,
            t=0.0,
            _diffusion=diffusion,
        )
        out_sqrt, _ = self.transition._backward_rv_sqrt(
            randvars.Constant(some_normal_rv1.mean),
            some_normal_rv2,
            rv_forwarded=rv_forward,
            gain=gain,
            t=0.0,
            _diffusion=diffusion,
        )
        out_joseph, _ = self.transition._backward_rv_joseph(
            randvars.Constant(some_normal_rv1.mean),
            some_normal_rv2,
            rv_forwarded=rv_forward,
            gain=gain,
            t=0.0,
            _diffusion=diffusion,
        )

        # Classic -- sqrt
        np.testing.assert_allclose(out_classic.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_classic.cov, out_sqrt.cov)

        # Joseph -- sqrt
        np.testing.assert_allclose(out_joseph.mean, out_sqrt.mean)
        np.testing.assert_allclose(out_joseph.cov, out_sqrt.cov)


class TestLinearGaussianLinOps:
    """Test class for linear Gaussians using LinearOperators where possible.

    Also tests that different forward and backward deal correctly when LinearOperators
    are used.
    """

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        state_dim,
        spdmat1,
        spdmat2,
    ):
        with config(matrix_free=True):
            self.noise_fun = lambda t: randvars.Normal(
                mean=np.arange(state_dim), cov=linops.aslinop(spdmat2)
            )
            self.transition_matrix_fun = lambda t: linops.aslinop(spdmat1)

            self.transition = randprocs.markov.discrete.LinearGaussian(
                input_dim=state_dim,
                output_dim=state_dim,
                transition_matrix_fun=self.transition_matrix_fun,
                noise_fun=self.noise_fun,
                forward_implementation="classic",
                backward_implementation="classic",
            )
            self.sqrt_transition = randprocs.markov.discrete.LinearGaussian(
                input_dim=state_dim,
                output_dim=state_dim,
                transition_matrix_fun=self.transition_matrix_fun,
                noise_fun=self.noise_fun,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            )

            self.transition_fun = lambda t, x: self.transition_matrix_fun(t) @ x
            self.transition_fun_jacobian = lambda t, x: self.transition_matrix_fun(t)

    # Test access to system matrices

    def test_transition_matrix_fun_fun(self):
        received = self.transition.transition_matrix_fun(0.0)
        expected = self.transition_matrix_fun(0.0)
        np.testing.assert_allclose(received.todense(), expected.todense())

    # Test forward and backward implementations

    def test_forward_rv(self, some_normal_rv1):
        array_cov_rv = some_normal_rv1
        linop_cov_rv = randvars.Normal(
            array_cov_rv.mean.copy(), linops.aslinop(array_cov_rv.cov)
        )
        with config(matrix_free=True):
            with pytest.warns(RuntimeWarning):
                self.transition.forward_rv(array_cov_rv, 0.0)

            out, _ = self.transition.forward_rv(linop_cov_rv, 0.0)
            assert isinstance(out, randvars.Normal)
            assert isinstance(out.cov, linops.LinearOperator)
            assert isinstance(out._cov_cholesky, linops.LinearOperator)

            with pytest.raises(NotImplementedError):
                self.sqrt_transition.forward_rv(array_cov_rv, 0.0)
            with pytest.raises(NotImplementedError):
                self.sqrt_transition.forward_rv(linop_cov_rv, 0.0)

    def test_backward_rv_classic(self, some_normal_rv1, some_normal_rv2):
        array_cov_rv1 = some_normal_rv1
        linop_cov_rv1 = randvars.Normal(
            array_cov_rv1.mean.copy(), linops.aslinop(array_cov_rv1.cov)
        )
        array_cov_rv2 = some_normal_rv2
        linop_cov_rv2 = randvars.Normal(
            array_cov_rv2.mean.copy(), linops.aslinop(array_cov_rv2.cov)
        )
        with config(matrix_free=True):
            with pytest.warns(RuntimeWarning):
                self.transition.backward_rv(array_cov_rv1, array_cov_rv2)
            with pytest.warns(RuntimeWarning):
                self.transition.backward_rv(linop_cov_rv1, array_cov_rv2)
            with pytest.warns(RuntimeWarning):
                self.transition.backward_rv(array_cov_rv1, linop_cov_rv2)

            out, _ = self.transition.backward_rv(linop_cov_rv1, linop_cov_rv2)
            assert isinstance(out, randvars.Normal)
            assert isinstance(out.cov, linops.LinearOperator)
            assert isinstance(out._cov_cholesky, linops.LinearOperator)

            with pytest.raises(NotImplementedError):
                self.sqrt_transition.backward_rv(array_cov_rv1, array_cov_rv2)
            with pytest.raises(NotImplementedError):
                self.sqrt_transition.backward_rv(linop_cov_rv1, linop_cov_rv2)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with config(matrix_free=True):
            array_cov_rv = some_normal_rv2
            linop_cov_rv = randvars.Normal(
                array_cov_rv.mean.copy(), linops.aslinop(array_cov_rv.cov)
            )
            with pytest.warns(RuntimeWarning):
                self.transition.backward_realization(some_normal_rv1.mean, array_cov_rv)
            out, _ = self.transition.backward_realization(
                some_normal_rv1.mean, linop_cov_rv
            )
            assert isinstance(out, randvars.Normal)
