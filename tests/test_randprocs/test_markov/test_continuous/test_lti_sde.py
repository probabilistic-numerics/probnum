import numpy as np
import pytest

from probnum import randprocs, randvars
from tests.test_randprocs.test_markov import test_transition


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
