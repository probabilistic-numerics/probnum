import numpy as np
import pytest

from probnum import randprocs, randvars
from tests.test_randprocs.test_markov import test_transition


class TestSDE(test_transition.InterfaceTestTransition):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1):

        self.g = lambda t, x: np.sin(x)
        self.L = lambda t: spdmat1
        self.dg = lambda t, x: np.cos(x)
        self.transition = randprocs.markov.continuous.SDE(
            test_ndim, self.g, self.L, self.dg
        )

    # Test access to system matrices

    def test_drift(self, some_normal_rv1):
        expected = self.g(0.0, some_normal_rv1.mean)
        received = self.transition.driftfun(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    def test_dispersionmatrix(self):
        expected = self.L(0.0)
        received = self.transition.dispmatfun(0.0)
        np.testing.assert_allclose(received, expected)

    def test_jacobfun(self, some_normal_rv1):
        expected = self.dg(0.0, some_normal_rv1.mean)
        received = self.transition.jacobfun(0.0, some_normal_rv1.mean)
        np.testing.assert_allclose(received, expected)

    # Test forward and backward implementations

    def test_forward_rv(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_rv(some_normal_rv1, 0.0, dt=0.1)

    def test_forward_realization(self, some_normal_rv1):
        with pytest.raises(NotImplementedError):
            self.transition.forward_realization(some_normal_rv1.mean, 0.0, dt=0.1)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_rv(some_normal_rv1, some_normal_rv2, 0.0, dt=0.1)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        with pytest.raises(NotImplementedError):
            self.transition.backward_realization(
                some_normal_rv1.mean, some_normal_rv2, 0.0, dt=0.1
            )

    def test_input_dim(self, test_ndim):
        assert self.transition.input_dim == test_ndim

    def test_output_dim(self, test_ndim):
        assert self.transition.output_dim == test_ndim

    def test_dimension(self, test_ndim):
        assert self.transition.dimension == test_ndim


class TestLinearSDE(TestSDE):

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, test_ndim, spdmat1, spdmat2):

        self.G = lambda t: spdmat1
        self.v = lambda t: np.arange(test_ndim)
        self.L = lambda t: spdmat2
        self.transition = randprocs.markov.continuous.LinearSDE(
            test_ndim, self.G, self.v, self.L
        )

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    def test_driftmatfun(self):
        expected = self.G(0.0)
        received = self.transition.driftmatfun(0.0)
        np.testing.assert_allclose(expected, received)

    def test_forcevecfun(self):
        expected = self.v(0.0)
        received = self.transition.forcevecfun(0.0)
        np.testing.assert_allclose(expected, received)

    def test_forward_rv(self, some_normal_rv1):
        out, _ = self.transition.forward_rv(some_normal_rv1, t=0.0, dt=0.1)
        assert isinstance(out, randvars.Normal)

    def test_forward_realization(self, some_normal_rv1):
        out, info = self.transition.forward_realization(
            some_normal_rv1.mean, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_rv(
            some_normal_rv1, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_realization(
            some_normal_rv1.mean, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

    def test_forward_realization_value_error_caught(self, some_normal_rv1):
        """the forward realization only works if a time-increment dt is provided."""
        with pytest.raises(ValueError):
            self.transition.forward_realization(some_normal_rv1.mean, t=0.0)

    def test_backward_realization_value_error_caught(
        self, some_normal_rv1, some_normal_rv2
    ):
        """the backward realization only works if a time-increment dt is provided."""
        with pytest.raises(ValueError):
            out, _ = self.transition.backward_realization(
                some_normal_rv1.mean,
                some_normal_rv2,
                t=0.0,
            )


class TestLTISDE(TestLinearSDE):

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
        self.v_const = np.arange(test_ndim)
        self.L_const = spdmat2

        self.transition = randprocs.markov.continuous.LTISDE(
            self.G_const,
            self.v_const,
            self.L_const,
            forward_implementation=forw_impl_string_linear_gauss,
            backward_implementation=backw_impl_string_linear_gauss,
        )

        self.G = lambda t: spdmat1
        self.v = lambda t: np.arange(test_ndim)
        self.L = lambda t: spdmat2

        self.g = lambda t, x: self.G(t) @ x + self.v(t)
        self.dg = lambda t, x: self.G(t)

    def test_discretise(self):
        out = self.transition.discretise(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.DiscreteLTIGaussian)

    def test_discretise_no_force(self):
        """LTISDE.discretise() works if there is zero force (there is an "if" in the
        fct)."""
        self.transition.forcevec = 0.0 * self.transition.forcevec
        assert (
            np.linalg.norm(self.transition.forcevecfun(0.0)) == 0.0
        )  # side quest/test
        out = self.transition.discretise(dt=0.1)
        assert isinstance(out, randprocs.markov.discrete.DiscreteLTIGaussian)

    def test_backward_rv(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_rv(
            some_normal_rv1, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)

    def test_backward_realization(self, some_normal_rv1, some_normal_rv2):
        out, _ = self.transition.backward_realization(
            some_normal_rv1.mean, some_normal_rv2, t=0.0, dt=0.1
        )
        assert isinstance(out, randvars.Normal)


@pytest.fixture
def G_const():
    return np.array([[0.0, 1.0], [0.0, 0.0]])


@pytest.fixture
def v_const():
    return np.array([1.0, 1.0])


@pytest.fixture
def L_const():
    return np.array([[0.0], [1.0]])


@pytest.fixture
def ltisde_as_linearsde(G_const, v_const, L_const):
    G = lambda t: G_const
    v = lambda t: v_const
    L = lambda t: L_const
    dim = 2

    return randprocs.markov.continuous.LinearSDE(
        dim, G, v, L, mde_atol=1e-12, mde_rtol=1e-12
    )


@pytest.fixture
def ltisde_as_linearsde_sqrt_forward_implementation(G_const, v_const, L_const):
    G = lambda t: G_const
    v = lambda t: v_const
    L = lambda t: L_const
    dim = 2

    return randprocs.markov.continuous.LinearSDE(
        dim, G, v, L, mde_atol=1e-12, mde_rtol=1e-12, forward_implementation="sqrt"
    )


@pytest.fixture
def ltisde(G_const, v_const, L_const):
    return randprocs.markov.continuous.LTISDE(G_const, v_const, L_const)


def test_solve_mde_forward_values(ltisde_as_linearsde, ltisde, v_const, diffusion):
    out_linear, _ = ltisde_as_linearsde.forward_realization(
        v_const, t=0.0, dt=0.1, _diffusion=diffusion
    )
    out_lti, _ = ltisde.forward_realization(
        v_const, t=0.0, dt=0.1, _diffusion=diffusion
    )

    np.testing.assert_allclose(out_linear.mean, out_lti.mean)
    np.testing.assert_allclose(out_linear.cov, out_lti.cov)


def test_solve_mde_forward_sqrt_values(
    ltisde_as_linearsde,
    ltisde_as_linearsde_sqrt_forward_implementation,
    v_const,
    diffusion,
):
    """mde forward values in sqrt-implementation and classic implementation should be
    equal."""
    out_linear, _ = ltisde_as_linearsde.forward_realization(
        v_const, t=0.0, dt=0.1, _diffusion=diffusion
    )

    out_linear_2, _ = ltisde_as_linearsde.forward_rv(
        out_linear, t=0.1, dt=0.1, _diffusion=diffusion
    )
    out_linear_2_sqrt, _ = ltisde_as_linearsde_sqrt_forward_implementation.forward_rv(
        out_linear, t=0.1, dt=0.1, _diffusion=diffusion
    )

    np.testing.assert_allclose(out_linear_2_sqrt.mean, out_linear_2.mean)
    np.testing.assert_allclose(out_linear_2_sqrt.cov, out_linear_2.cov)


def test_solve_mde_backward_values(ltisde_as_linearsde, ltisde, v_const, diffusion):
    out_linear_forward, _ = ltisde_as_linearsde.forward_realization(
        v_const, t=0.0, dt=0.1, _diffusion=diffusion
    )
    out_lti_forward, _ = ltisde.forward_realization(
        v_const, t=0.0, dt=0.1, _diffusion=diffusion
    )
    out_linear_forward_next, _ = ltisde_as_linearsde.forward_rv(
        out_linear_forward, t=0.1, dt=0.1, _diffusion=diffusion
    )
    out_lti_forward_next, _ = ltisde.forward_rv(
        out_lti_forward, t=0.1, dt=0.1, _diffusion=diffusion
    )

    out_linear, _ = ltisde_as_linearsde.backward_realization(
        realization_obtained=out_linear_forward_next.mean,
        rv=out_linear_forward,
        t=0.1,
        dt=0.1,
        _diffusion=diffusion,
    )
    out_lti, _ = ltisde.backward_realization(
        realization_obtained=out_lti_forward_next.mean,
        rv=out_lti_forward,
        t=0.1,
        dt=0.1,
        _diffusion=diffusion,
    )

    np.testing.assert_allclose(out_linear.mean, out_lti.mean)
    np.testing.assert_allclose(out_linear.cov, out_lti.cov, atol=1e-9)
