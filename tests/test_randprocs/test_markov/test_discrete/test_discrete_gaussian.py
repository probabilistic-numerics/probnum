import numpy as np
import pytest

from probnum import config, linops, randprocs, randvars
from tests.test_randprocs.test_markov import test_transition


class TestDiscreteGaussian(test_transition.InterfaceTestTransition):
    """Tests for the discrete Gaussian class.

    Most of the tests are reused/overwritten in subclasses, therefore the class-
    structure. Unfortunately, testing the product space of all combinations (different
    subclasses, different implementations, etc.) generates a lot of tests that are
    executed.
    """

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1):

        self.g = lambda t, x: np.sin(x)
        self.S = lambda t: spdmat1
        self.dg = lambda t, x: np.cos(x)
        self.transition = randprocs.markov.discrete.DiscreteGaussian(
            test_ndim,
            test_ndim,
            self.g,
            self.S,
            self.dg,
            None,
        )

    # Test access to system matrices

    def test_state_transition(self, some_normal_rv1):
        received = self.transition.state_trans_fun(0.0, some_normal_rv1.mean)
        expected = self.g(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_process_noise(self):
        received = self.transition.proc_noise_cov_mat_fun(0.0)
        expected = self.S(0.0)
        np.testing.assert_allclose(received, expected)

    def test_process_noise_cholesky(self):
        received = self.transition.proc_noise_cov_cholesky_fun(0.0)
        expected = np.linalg.cholesky(self.transition.proc_noise_cov_mat_fun(0.0))
        np.testing.assert_allclose(received, expected)

    def test_jacobian(self, some_normal_rv1):
        received = self.transition.jacob_state_trans_fun(0.0, some_normal_rv1.mean)
        expected = self.dg(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    # Test forward and backward implementations

    def test_forward_rv(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_rv(some_normal_rv1, 0.0)

    def test_forward_realization(self, some_normal_rv1):
        out, _ = self.transition.forward_realization(some_normal_rv1.mean, 0.0)
        assert isinstance(out, randvars.Normal)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_rv(some_normal_rv1, some_normal_rv2)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_realization(some_normal_rv1.mean, some_normal_rv2)

    def test_input_dim(self, test_ndim):
        assert self.transition.input_dim == test_ndim

    def test_output_dim(self, test_ndim):
        assert self.transition.output_dim == test_ndim


@pytest.fixture(params=["classic", "sqrt"])
def forw_impl_string_linear_gauss(request):
    """Forward implementation choices passed via strings."""
    return request.param


@pytest.fixture(params=["classic", "joseph", "sqrt"])
def backw_impl_string_linear_gauss(request):
    """Backward implementation choices passed via strings."""
    return request.param


class TestLinearGaussian(TestDiscreteGaussian):
    """Test class for linear Gaussians. Inherits tests from `TestDiscreteGaussian` but
    overwrites the forward and backward transitions.

    Also tests that different forward and backward implementations yield the same
    results.
    """

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        test_ndim,
        spdmat1,
        spdmat2,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):

        self.G = lambda t: spdmat1
        self.S = lambda t: spdmat2
        self.v = lambda t: np.arange(test_ndim)
        self.transition = randprocs.markov.discrete.DiscreteLinearGaussian(
            test_ndim,
            test_ndim,
            self.G,
            self.v,
            self.S,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    # Test access to system matrices

    def test_state_transition_mat_fun(self):
        received = self.transition.state_trans_mat_fun(0.0)
        expected = self.G(0.0)
        np.testing.assert_allclose(received, expected)

    def test_shift_vec_fun(self):
        received = self.transition.shift_vec_fun(0.0)
        expected = self.v(0.0)
        np.testing.assert_allclose(received, expected)

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


class TestLTIGaussian(TestLinearGaussian):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(
        self,
        test_ndim,
        spdmat1,
        spdmat2,
        forw_impl_string_linear_gauss,
        backw_impl_string_linear_gauss,
    ):

        self.G_const = spdmat1
        self.S_const = spdmat2
        self.v_const = np.arange(test_ndim)
        self.transition = randprocs.markov.discrete.DiscreteLTIGaussian(
            self.G_const,
            self.v_const,
            self.S_const,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        # Compatibility with superclass' test
        self.G = lambda t: self.G_const
        self.S = lambda t: self.S_const
        self.v = lambda t: self.v_const
        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    # Test access to system matrices

    def test_state_transition_mat(self):
        received = self.transition.state_trans_mat
        expected = self.G_const
        np.testing.assert_allclose(received, expected)

    def test_shift_vec(self):
        received = self.transition.shift_vec
        expected = self.v_const
        np.testing.assert_allclose(received, expected)

    def test_process_noise_cov_mat(self):
        received = self.transition.proc_noise_cov_mat
        expected = self.S_const
        np.testing.assert_allclose(received, expected)

    def test_process_noise_cov_cholesky(self):
        received = self.transition.proc_noise_cov_cholesky
        expected = np.linalg.cholesky(self.S_const)
        np.testing.assert_allclose(received, expected)


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
        test_ndim,
        spdmat1,
        spdmat2,
    ):
        with config(lazy_linalg=True):
            self.G = lambda t: linops.aslinop(spdmat1)
            self.S = lambda t: linops.aslinop(spdmat2)
            self.v = lambda t: np.arange(test_ndim)
            self.transition = randprocs.markov.discrete.DiscreteLinearGaussian(
                test_ndim,
                test_ndim,
                self.G,
                self.v,
                self.S,
                forward_implementation="classic",
                backward_implementation="classic",
            )
            self.sqrt_transition = randprocs.markov.discrete.DiscreteLinearGaussian(
                test_ndim,
                test_ndim,
                self.G,
                self.v,
                self.S,
                forward_implementation="sqrt",
                backward_implementation="sqrt",
            )

            self.g = lambda t, x: self.G(t) @ x + self.v(t)
            self.dg = lambda t, x: self.G(t)

    # Test access to system matrices

    def test_state_transition_mat_fun(self):
        received = self.transition.state_trans_mat_fun(0.0)
        expected = self.G(0.0)
        np.testing.assert_allclose(received.todense(), expected.todense())

    def test_shift_vec_fun(self):
        received = self.transition.shift_vec_fun(0.0)
        expected = self.v(0.0)
        np.testing.assert_allclose(received, expected)

    # Test forward and backward implementations

    def test_forward_rv(self, some_normal_rv1):
        array_cov_rv = some_normal_rv1
        linop_cov_rv = randvars.Normal(
            array_cov_rv.mean.copy(), linops.aslinop(array_cov_rv.cov)
        )
        with config(lazy_linalg=True):
            with pytest.warns(RuntimeWarning):
                self.transition.forward_rv(array_cov_rv, 0.0)

            out, _ = self.transition.forward_rv(linop_cov_rv, 0.0)
            assert isinstance(out, randvars.Normal)
            assert isinstance(out.cov, linops.LinearOperator)
            assert isinstance(out.cov_cholesky, linops.LinearOperator)

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
        with config(lazy_linalg=True):
            with pytest.warns(RuntimeWarning):
                self.transition.backward_rv(array_cov_rv1, array_cov_rv2)
            with pytest.warns(RuntimeWarning):
                self.transition.backward_rv(linop_cov_rv1, array_cov_rv2)
            with pytest.warns(RuntimeWarning):
                self.transition.backward_rv(array_cov_rv1, linop_cov_rv2)

            out, _ = self.transition.backward_rv(linop_cov_rv1, linop_cov_rv2)
            assert isinstance(out, randvars.Normal)
            assert isinstance(out.cov, linops.LinearOperator)
            assert isinstance(out.cov_cholesky, linops.LinearOperator)

            with pytest.raises(NotImplementedError):
                self.sqrt_transition.backward_rv(array_cov_rv1, array_cov_rv2)
            with pytest.raises(NotImplementedError):
                self.sqrt_transition.backward_rv(linop_cov_rv1, linop_cov_rv2)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with config(lazy_linalg=True):
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
